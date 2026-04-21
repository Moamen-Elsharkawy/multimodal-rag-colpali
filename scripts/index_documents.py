"""
scripts/index_documents.py

Builds the FAISS index from a directory of PDFs and saves it to disk.
This is the offline step — run it once, then the Streamlit app loads
the pre-built index instantly without needing to re-embed everything.

Run:
    python scripts/index_documents.py --data_dir data/ --index_dir index/

Optional flags:
    --model   override the ColPali model (default: vidore/colpali-v1.3)
    --dpi     rendering resolution (default: 150)
    --batch   embedding batch size (default: 4, lower if OOM)
    --limit   max pages per PDF (useful for quick testing)
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))


def _is_real_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def main(
    data_dir: str,
    index_dir: str,
    model_name: str,
    dpi: int,
    batch_size: int,
    max_pages_per_pdf: int = None,
):
    from src.ingestion.pdf_processor import PDFProcessor
    from src.ingestion.indexer import DocumentIndex, PageMeta
    from src.retrieval.text_retriever import TextPageIndex

    data_dir  = Path(data_dir)
    index_dir = Path(index_dir)
    img_dir   = index_dir / "page_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(p for p in data_dir.glob("*.pdf") if _is_real_pdf(p))
    if not pdf_files:
        logger.error(f"No PDFs found in {data_dir}. Run download_dataset.py first.")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")

    # Check if TF-IDF-only mode is forced
    use_tfidf_only = os.getenv("USE_TFIDF_ONLY", "").lower() in ("true", "1", "yes")
    
    # load models
    processor = PDFProcessor(dpi=dpi, max_pages=max_pages_per_pdf)
    text_fallback = False

    if use_tfidf_only:
        logger.warning("USE_TFIDF_ONLY is enabled. Building TF-IDF-only index (skipping ColPali).")
        embedder = None
        index = TextPageIndex()
        text_fallback = True
    else:
        try:
            from src.ingestion.embedder import ColPaliEmbedder

            logger.info(f"Loading ColPali model: {model_name}")
            embedder = ColPaliEmbedder(model_name=model_name, batch_size=batch_size)
            index = DocumentIndex(embed_dim=128)
        except Exception as exc:
            logger.warning(f"ColPali could not be loaded: {exc}")
            logger.warning("Falling back to a local TF-IDF page retriever for this build.")
            logger.info("💡 Tip: To skip ColPali in the future, set USE_TFIDF_ONLY=true in your .env file")
            embedder = None
            index = TextPageIndex()
            text_fallback = True

    page_id_counter = 0
    all_metas: list = []
    all_embs: list = []
    all_texts: list[str] = []

    for pdf_file in pdf_files:
        logger.info(f"\nProcessing: {pdf_file.name}")

        # Step 1: PDF → images
        pages = processor.process_pdf(str(pdf_file))
        logger.info(f"  Rendered {len(pages)} pages at {dpi} DPI")

        # Step 2: save page images to disk (needed by retriever to load at query time)
        saved_imgs = []
        for page in pages:
            img_path = img_dir / f"{page.doc_name}_p{page.page_number:04d}.png"
            if not img_path.exists():
                page.image.save(str(img_path), format="PNG", optimize=True)
            saved_imgs.append(str(img_path))

        embs = None
        if not text_fallback:
            # Step 3: embed pages in batches
            logger.info(f"  Embedding pages (batch_size={batch_size}) ...")
            images_list = [p.image for p in pages]
            embs = embedder.embed_images(images_list)

        # Step 4: build metadata + register with index
        for idx_in_pdf, page in enumerate(pages):
            img_path = saved_imgs[idx_in_pdf]
            meta = PageMeta(
                page_id=page_id_counter,
                doc_name=page.doc_name,
                doc_path=str(pdf_file.resolve()),
                page_number=page.page_number,
                image_path=img_path,
            )
            all_metas.append(meta)
            if text_fallback:
                all_texts.append(page.text)
            else:
                all_embs.append(embs[idx_in_pdf])
            page_id_counter += 1

        logger.info(f"  Done — cumulative: {page_id_counter} pages indexed")

    # Step 5: build FAISS index and save
    logger.info(f"\nSaving index to {index_dir} ...")
    if text_fallback:
        index.add_pages(all_metas, all_texts)
        index.save(str(index_dir))
        logger.info("\n" + "=" * 55)
        logger.info("✅ Indexing complete with TF-IDF fallback")
        logger.info(f"   Total pages: {page_id_counter}")
        logger.info(f"   Index dir  : {index_dir}")
    else:
        logger.info(f"Adding {len(all_metas)} pages to FAISS index ...")
        index.add_pages(all_metas, all_embs)
        index.save(str(index_dir))
        logger.info("\n" + "=" * 55)
        logger.info("✅ Indexing complete with ColPali")
        logger.info(f"   Total pages  : {page_id_counter}")
        logger.info(f"   Index dir    : {index_dir}")
        logger.info(f"   Patch vectors: {index.faiss_index.ntotal:,}")

    logger.info(f"\nNow run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ColPali FAISS index from PDFs")
    parser.add_argument("--data_dir",  default="data/",              help="Directory with PDFs")
    parser.add_argument("--index_dir", default="index/",             help="Output index directory")
    parser.add_argument(
        "--model",
        default=os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3"),
        help="ColPali model name or a local checkpoint path",
    )
    parser.add_argument("--dpi",       type=int, default=150,         help="Page render DPI")
    parser.add_argument("--batch",     type=int, default=4,           help="Embedding batch size")
    parser.add_argument("--limit",     type=int, default=None,        help="Max pages per PDF")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        model_name=args.model,
        dpi=args.dpi,
        batch_size=args.batch,
        max_pages_per_pdf=args.limit,
    )
