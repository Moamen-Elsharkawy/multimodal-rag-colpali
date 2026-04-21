"""
app/streamlit_app.py

The main demo interface. Run with:
    streamlit run app/streamlit_app.py

Features:
  - PDF upload (processes pages on the fly) or load from a saved index
  - Interactive question input with chat history
  - Displays the retrieved pages with citations and scores
  - Supports OpenAI, OpenRouter, or a local Qwen2-VL generator
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Make sure the project root is available regardless of where Streamlit starts.
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()


def _default_colpali_model() -> str:
    return os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3")


def _generator_mode_value(label: str) -> str:
    return "local" if label.startswith("Local") else "auto"


def _format_runtime_error(exc: Exception) -> str:
    message = str(exc)

    if "Failed to load the ColPali model" in message or "ColPali model loading is disabled" in message:
        return (
            f"{message}\n\n"
            "**Fallback to TF-IDF**: The app will automatically use TF-IDF-based retrieval, "
            "which is slower but still functional without Hugging Face access.\n\n"
            "**Options**:\n"
            "1. If you have a local model checkpoint, provide its path in the config\n"
            "2. Set `USE_TFIDF_ONLY=true` in your `.env` file to skip ColPali entirely\n"
            "3. Wait until you have network access to huggingface.co to cache the model"
        )

    if "OPENAI_API_KEY" in message or "OPENROUTER_API_KEY" in message:
        return (
            f"{message}\n\n"
            "If OpenAI is unavailable, add OPENROUTER_API_KEY and OPENROUTER_MODEL "
            "to your .env file and keep the generator mode on Auto."
        )

    return message


def _render_retrieved_pages(page_entries: list[dict]):
    if not page_entries:
        return

    page_images = st.session_state.page_images
    cols = st.columns(len(page_entries))
    for col, entry in zip(cols, page_entries):
        with col:
            image = page_images.get(entry["page_id"])
            if image is not None:
                st.image(image, use_container_width=True)
            else:
                st.info("Image unavailable")
            st.caption(f"{entry['citation']}\nScore: {entry['score']:.2f}")


@st.cache_resource(show_spinner="Loading ColPali model ...")
def get_embedder(model_name: str):
    from src.ingestion.embedder import ColPaliEmbedder

    return ColPaliEmbedder(model_name=model_name, batch_size=2)


@st.cache_resource(show_spinner="Loading answer generator ...")
def get_generator(mode: str):
    from src.generation.generator import AnswerGenerator

    return AnswerGenerator(use_local=(mode == "local"), provider=mode)


def build_index_from_uploads(uploaded_files, dpi: int, embedder):
    """Process uploaded PDFs and build an in-memory index."""
    from src.ingestion.indexer import DocumentIndex, PageMeta
    from src.ingestion.pdf_processor import PDFProcessor

    processor = PDFProcessor(dpi=dpi)
    index = DocumentIndex(embed_dim=128)
    page_images = {}
    all_metas = []
    all_embs = []
    page_id_counter = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        img_dir = tmpdir_path / "pages"
        img_dir.mkdir()

        progress = st.progress(0, text="Processing PDFs ...")

        for file_idx, uploaded_file in enumerate(uploaded_files):
            pdf_path = tmpdir_path / uploaded_file.name
            pdf_path.write_bytes(uploaded_file.read())

            pages = processor.process_pdf(str(pdf_path))
            progress.progress(
                (file_idx / max(len(uploaded_files), 1)) * 0.5,
                text=f"Rendering {uploaded_file.name} ({len(pages)} pages)",
            )

            embeddings = embedder.embed_images([page.image for page in pages])

            for page, emb in zip(pages, embeddings):
                img_path = img_dir / f"{page.doc_name}_p{page.page_number:04d}.png"
                page.image.save(str(img_path), format="PNG")

                meta = PageMeta(
                    page_id=page_id_counter,
                    doc_name=page.doc_name,
                    doc_path=page.doc_path,
                    page_number=page.page_number,
                    image_path=str(img_path),
                )
                page_images[page_id_counter] = page.image.copy()
                all_metas.append(meta)
                all_embs.append(emb)
                page_id_counter += 1

        progress.progress(0.85, text="Building FAISS index ...")
        index.add_pages(all_metas, all_embs)
        progress.progress(1.0, text="Index ready")

    return index, page_images


def build_text_fallback_from_uploads(uploaded_files, dpi: int, top_k: int):
    """Build an in-memory TF-IDF fallback retriever from uploaded PDFs."""
    from src.ingestion.indexer import PageMeta
    from src.ingestion.pdf_processor import PDFProcessor
    from src.retrieval.text_retriever import TextPageIndex, TextPageRetriever

    processor = PDFProcessor(dpi=dpi)
    index = TextPageIndex()
    page_images = {}
    all_metas = []
    all_texts = []
    page_id_counter = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        img_dir = tmpdir_path / "pages"
        img_dir.mkdir()

        progress = st.progress(0, text="Processing PDFs with text fallback ...")

        for file_idx, uploaded_file in enumerate(uploaded_files):
            pdf_path = tmpdir_path / uploaded_file.name
            pdf_path.write_bytes(uploaded_file.read())

            pages = processor.process_pdf(str(pdf_path))
            progress.progress(
                (file_idx / max(len(uploaded_files), 1)) * 0.7,
                text=f"Extracting page text from {uploaded_file.name} ({len(pages)} pages)",
            )

            for page in pages:
                img_path = img_dir / f"{page.doc_name}_p{page.page_number:04d}.png"
                page.image.save(str(img_path), format="PNG")

                meta = PageMeta(
                    page_id=page_id_counter,
                    doc_name=page.doc_name,
                    doc_path=page.doc_path,
                    page_number=page.page_number,
                    image_path=str(img_path),
                )
                page_images[page_id_counter] = page.image.copy()
                all_metas.append(meta)
                all_texts.append(page.text)
                page_id_counter += 1

        progress.progress(0.9, text="Building TF-IDF fallback index ...")
        index.add_pages(all_metas, all_texts)
        progress.progress(1.0, text="Text fallback ready")

    return TextPageRetriever(index=index, top_k=top_k), page_images


def load_existing_index(index_dir: str, model_name: str, top_k: int):
    """Load a saved index and reconstruct page images from disk."""
    from src.ingestion.indexer import DocumentIndex
    from src.retrieval.retriever import ColPaliRetriever
    from src.retrieval.text_retriever import TextPageIndex, TextPageRetriever

    if TextPageIndex.can_load(index_dir):
        index = TextPageIndex.load(index_dir)
        retriever = TextPageRetriever(index=index, top_k=top_k)
        retrieval_mode = "text"
    else:
        index = DocumentIndex.load(index_dir)
        retriever = ColPaliRetriever(
            index=index,
            embedder=get_embedder(model_name),
            top_k=top_k,
        )
        retrieval_mode = "colpali"

    page_images = {}
    for page_id, meta in index.page_metas.items():
        try:
            with Image.open(meta.image_path) as img:
                page_images[page_id] = img.convert("RGB").copy()
        except Exception:
            page_images[page_id] = Image.new("RGB", (200, 280), color=(240, 240, 240))

    return retriever, page_images, retrieval_mode


st.set_page_config(
    page_title="Multi-Modal RAG | ColPali",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 Multi-Modal Document Intelligence")

# Check if using TF-IDF-only mode and update subtitle
if os.getenv("USE_TFIDF_ONLY", "").lower() in ("true", "1", "yes"):
    st.caption("🔄 **TF-IDF-based retrieval** (no HuggingFace required) - Works offline!")
else:
    st.caption("ColPali-based retrieval over full PDF page images, with citation-backed answers.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "page_images" not in st.session_state:
    st.session_state.page_images = {}
if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = None

with st.sidebar:
    st.header("Configuration")

    # Determine default retrieval mode from environment
    default_tfidf_only = os.getenv("USE_TFIDF_ONLY", "").lower() in ("true", "1", "yes")
    default_index = 1 if default_tfidf_only else 0  # 1 = TF-IDF, 0 = Auto

    retrieval_mode_label = st.radio(
        "Retrieval mode",
        options=["Auto (ColPali if available)", "Force TF-IDF only"],
        index=default_index,
        help="Auto: Try ColPali first, fallback to TF-IDF if Hugging Face is unreachable.\n\n"
             "Force TF-IDF: Skip ColPali entirely and use keyword-based TF-IDF retrieval (faster, reliable without HuggingFace).",
    )
    force_tfidf_only = retrieval_mode_label.startswith("Force TF-IDF")
    
    if force_tfidf_only:
        os.environ["USE_TFIDF_ONLY"] = "true"
        if default_tfidf_only:
            st.info("🔄 **TF-IDF-only mode** (default) — No HuggingFace access required!")
        else:
            st.info("🔄 TF-IDF-only mode enabled — ColPali model will be skipped.")
    else:
        os.environ.pop("USE_TFIDF_ONLY", None)
        st.info("⚙️ Auto mode: Will try ColPali, fallback to TF-IDF if HuggingFace is unreachable")

    colpali_model = st.text_input(
        "ColPali / ColQwen model or local path",
        value=_default_colpali_model(),
        help="Use a Hugging Face model ID (e.g., 'vidore/colpali-v1.3') or a local checkpoint path.\n\nIgnored if using TF-IDF mode.",
        disabled=force_tfidf_only,
    )

    top_k = st.slider("Pages to retrieve (top-k)", min_value=1, max_value=6, value=3)
    dpi = st.slider(
        "Rendering DPI",
        min_value=72,
        max_value=200,
        value=120,
        help="Higher DPI preserves more detail, but increases memory use.",
    )

    generator_label = st.radio(
        "Answer generation",
        options=["Auto (OpenAI/OpenRouter)", "Local Qwen2-VL-2B"],
        index=0,
    )
    generator_mode = _generator_mode_value(generator_label)

    if generator_mode == "auto" and not (
        os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    ):
        st.info(
            "No remote API key found. Set OPENAI_API_KEY, or if that fails use "
            "OPENROUTER_API_KEY and OPENROUTER_MODEL in .env."
        )

    st.divider()
    st.header("Document Source")

    source_mode = st.radio(
        "Choose source",
        options=["Upload PDF(s)", "Load pre-built index"],
        index=0,
    )

    uploaded_files = None
    index_dir = None

    if source_mode == "Upload PDF(s)":
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
    else:
        index_dir = st.text_input("Index directory", value="index/")

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

build_button = st.button("Build / Load Index", type="primary")

if build_button:
    try:
        with st.spinner("Preparing index ..."):
            if source_mode == "Upload PDF(s)" and uploaded_files:
                try:
                    embedder = get_embedder(colpali_model)
                    index, page_images = build_index_from_uploads(uploaded_files, dpi, embedder)

                    from src.retrieval.retriever import ColPaliRetriever

                    retriever = ColPaliRetriever(
                        index=index,
                        embedder=embedder,
                        top_k=top_k,
                    )
                    retrieval_mode = "colpali"
                except Exception as exc:
                    st.warning(
                        _format_runtime_error(exc)
                        + "\n\nFalling back to page-level TF-IDF retrieval so the app remains usable."
                    )
                    retriever, page_images = build_text_fallback_from_uploads(
                        uploaded_files,
                        dpi,
                        top_k,
                    )
                    retrieval_mode = "text"
            elif source_mode == "Load pre-built index" and index_dir:
                retriever, page_images, retrieval_mode = load_existing_index(
                    index_dir,
                    colpali_model,
                    top_k,
                )
            else:
                st.error("Please upload PDFs or specify an index directory first.")
                st.stop()

        st.session_state.retriever = retriever
        st.session_state.page_images = page_images
        st.session_state.retrieval_mode = retrieval_mode
        st.session_state.index_ready = True
        
        if retrieval_mode == "colpali":
            st.success(f"✅ Index ready: {len(page_images)} pages using **ColPali** retrieval")
        else:
            st.success(f"✅ Index ready: {len(page_images)} pages using **TF-IDF** fallback retrieval")
    except Exception as exc:
        st.session_state.index_ready = False
        st.error(_format_runtime_error(exc))

if st.session_state.index_ready:
    st.divider()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("pages"):
                _render_retrieved_pages(msg["pages"])

    query = st.chat_input("Ask a question about your documents ...")

    if query:
        retriever = st.session_state.retriever

        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant pages ..."):
                results = retriever.retrieve(query)

            page_entries = []
            retrieved_images = []
            page_citations = []

            for result in results:
                image = st.session_state.page_images.get(result.page_id)
                if image is None:
                    continue

                page_entries.append(
                    {
                        "page_id": result.page_id,
                        "citation": result.citation,
                        "score": float(result.score),
                    }
                )
                retrieved_images.append(image)
                page_citations.append(result.citation)

            st.markdown("**Retrieved pages**")
            _render_retrieved_pages(page_entries)

            if not retrieved_images:
                answer = "I couldn't retrieve any page images for this question."
            else:
                try:
                    generator = get_generator(generator_mode)
                    with st.spinner("Generating answer ..."):
                        answer = generator.generate(
                            question=query,
                            retrieved_images=retrieved_images,
                            page_citations=page_citations,
                        )
                except Exception as exc:
                    answer = (
                        "Generation is currently unavailable.\n\n"
                        f"{_format_runtime_error(exc)}"
                    )

            st.divider()
            st.markdown("**Answer**")
            st.markdown(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "pages": page_entries,
            }
        )

    if st.session_state.chat_history:
        st.divider()
        chat_json = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            label="Export chat as JSON",
            data=chat_json,
            file_name="chat_history.json",
            mime="application/json",
        )
else:
    st.info("Upload PDFs or load a saved index, then click **Build / Load Index**.")
    st.markdown(
        """
        ### What this system can do

        | Capability | Details |
        |---|---|
        | Text queries | Finds and reads paragraphs, footnotes, and captions |
        | Chart queries | Describes trends and values in figures |
        | Table queries | Extracts values from dense statistical pages |
        | Mixed queries | Handles pages that combine text and visuals |
        | Citations | Every answer references the retrieved page(s) |
        """
    )
