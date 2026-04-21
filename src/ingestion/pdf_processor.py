"""
ingestion/pdf_processor.py

Handles the first step of the pipeline: turning raw PDFs into page images.
Using PyMuPDF (fitz) here because it's significantly faster than pdf2image
on large documents and gives finer control over rendering DPI.

Each page is rendered at 150 DPI by default — high enough for ColPali to
pick up chart details and table text, but not so high that we balloon memory
for 200-page WHO reports.
"""

import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
from loguru import logger


@dataclass
class PageRecord:
    """Holds one rendered page alongside its source metadata."""
    doc_name: str          # filename without extension
    doc_path: str          # absolute path to the source PDF
    page_number: int       # 0-indexed
    image: Image.Image     # the rendered PIL image
    width: int
    height: int
    text: str = ""
    # populated later by the embedder
    embedding: Optional[object] = field(default=None, repr=False)


class PDFProcessor:
    """
    Converts PDF files into lists of PageRecord objects.

    Usage:
        processor = PDFProcessor(dpi=150)
        pages = processor.process_pdf("who_health_stats_2023.pdf")
        # returns List[PageRecord]
    """

    def __init__(self, dpi: int = 150, max_pages: Optional[int] = None):
        self.dpi = dpi
        self.max_pages = max_pages
        # fitz uses a matrix scale factor, not DPI directly
        self._scale = dpi / 72.0

    def process_pdf(self, pdf_path: str) -> List[PageRecord]:
        """
        Render all pages of a PDF and return them as PageRecord objects.

        Args:
            pdf_path: path to the PDF file

        Returns:
            list of PageRecord, one per page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        with open(pdf_path, "rb") as f:
            if f.read(5) != b"%PDF-":
                raise ValueError(f"Not a valid PDF file: {pdf_path}")

        doc_name = pdf_path.stem
        pages = []

        doc = fitz.open(str(pdf_path))
        total = len(doc)
        limit = min(total, self.max_pages) if self.max_pages else total
        logger.info(f"Rendering {limit}/{total} pages from '{doc_name}' at {self.dpi} DPI")

        mat = fitz.Matrix(self._scale, self._scale)

        for page_idx in range(limit):
            page = doc[page_idx]
            # render to a pixel map, then convert to PIL
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            page_text = page.get_text("text").strip()

            pages.append(PageRecord(
                doc_name=doc_name,
                doc_path=str(pdf_path.resolve()),
                page_number=page_idx,
                image=img,
                width=img.width,
                height=img.height,
                text=page_text,
            ))

        doc.close()
        logger.info(f"Done — {len(pages)} pages rendered")
        return pages

    def process_directory(self, dir_path: str) -> List[PageRecord]:
        """
        Process every PDF in a directory and concatenate all pages.

        Args:
            dir_path: path to folder containing PDFs

        Returns:
            flat list of all PageRecord objects across all documents
        """
        dir_path = Path(dir_path)
        pdf_files = sorted(dir_path.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDFs found in {dir_path}")

        logger.info(f"Found {len(pdf_files)} PDF(s) in {dir_path}")
        all_pages: List[PageRecord] = []

        for pdf_file in pdf_files:
            try:
                pages = self.process_pdf(str(pdf_file))
                all_pages.extend(pages)
            except Exception as e:
                logger.warning(f"Skipping {pdf_file.name}: {e}")

        logger.info(f"Total pages processed: {len(all_pages)}")
        return all_pages
