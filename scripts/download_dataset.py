"""
scripts/download_dataset.py

Downloads the WHO World Health Statistics reports used by this project.

The original version of this script pointed at stale bitstream URLs and would
silently save HTML landing pages with a ".pdf" extension. This implementation
resolves the current PDF asset from the WHO IRIS handle page and validates that
the downloaded bytes are actually a PDF before keeping the file on disk.

Run:
    python scripts/download_dataset.py --out_dir data/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import requests
import fitz
from loguru import logger
from tqdm import tqdm


USER_AGENT = "Mozilla/5.0 (research/educational use; multimodal-rag-colpali)"
PDF_HEADER = b"%PDF-"

# WHO IRIS handle pages are stable, while direct bitstream URLs change over time.
DATASETS = [
    {
        "name": "WHO World Health Statistics 2024",
        "handle_url": "https://iris.who.int/handle/10665/376869",
        "filename": "WHO_World_Health_Statistics_2024.pdf",
    },
    {
        "name": "WHO World Health Statistics 2023",
        "handle_url": "https://iris.who.int/handle/10665/367912",
        "filename": "WHO_World_Health_Statistics_2023.pdf",
    },
    {
        "name": "WHO World Health Statistics 2022",
        "handle_url": "https://iris.who.int/handle/10665/356584",
        "filename": "WHO_World_Health_Statistics_2022.pdf",
    },
]

# Keep the default dataset focused and reproducible for the assignment.
FALLBACK_DATASETS: list[dict] = []


def _is_pdf_response(response: requests.Response, first_chunk: bytes) -> bool:
    """Check the content type and file signature before trusting a response."""
    content_type = (response.headers.get("content-type") or "").lower()
    return "application/pdf" in content_type or first_chunk.startswith(PDF_HEADER)


def _is_pdf_file(path: Path) -> bool:
    """Validate an already-downloaded file by its magic bytes."""
    if not path.exists() or path.stat().st_size < len(PDF_HEADER):
        return False
    with open(path, "rb") as f:
        if f.read(len(PDF_HEADER)) != PDF_HEADER:
            return False

    try:
        with fitz.open(str(path)) as doc:
            return len(doc) > 0
    except Exception:
        return False


def _dedupe_keep_order(urls: Iterable[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _extract_candidate_urls(handle_url: str) -> list[str]:
    """Pull possible PDF asset URLs from a WHO IRIS handle page."""
    response = requests.get(
        handle_url,
        timeout=60,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    html = response.text

    candidates = []
    candidates.extend(
        re.findall(
            r"https://iris\.who\.int/server/api/core/bitstreams/[a-f0-9-]+/content",
            html,
            re.IGNORECASE,
        )
    )
    candidates.extend(
        re.findall(
            r"https://iris\.who\.int/bitstreams/[a-f0-9-]+/download",
            html,
            re.IGNORECASE,
        )
    )

    return _dedupe_keep_order(candidates)


def resolve_pdf_url(item: dict) -> str:
    """
    Resolve a dataset entry to a working PDF URL.

    For WHO IRIS handles we probe all candidate asset links and keep the first
    one that genuinely responds with PDF bytes.
    """
    if "url" in item:
        return item["url"]

    handle_url = item["handle_url"]
    candidates = _extract_candidate_urls(handle_url)

    if not candidates:
        raise RuntimeError(f"No candidate download links found on handle page: {handle_url}")

    for candidate in candidates:
        try:
            with requests.get(
                candidate,
                stream=True,
                timeout=120,
                headers={"User-Agent": USER_AGENT},
            ) as response:
                response.raise_for_status()
                first_chunk = next(response.iter_content(chunk_size=16), b"")
                if _is_pdf_response(response, first_chunk):
                    return candidate
                logger.debug(
                    "Skipping non-PDF candidate for {}: {} ({})",
                    item["name"],
                    candidate,
                    response.headers.get("content-type"),
                )
        except Exception as exc:
            logger.debug(f"Candidate probe failed for {candidate}: {exc}")

    raise RuntimeError(
        f"Could not resolve a valid PDF asset for {item['name']} from {handle_url}"
    )


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Stream-download a file with validation and a progress bar."""
    try:
        response = requests.get(
            url,
            timeout=300,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()

        data = response.content
        first_chunk = data[:chunk_size]
        if not _is_pdf_response(response, first_chunk):
            raise ValueError(
                f"Expected a PDF but received {response.headers.get('content-type')}"
            )

        with open(dest, "wb") as f, tqdm(
            total=len(data),
            unit="B",
            unit_scale=True,
            desc=dest.name,
            ncols=80,
        ) as bar:
            for offset in range(0, len(data), chunk_size):
                chunk = data[offset: offset + chunk_size]
                f.write(chunk)
                bar.update(len(chunk))

        if not _is_pdf_file(dest):
            raise ValueError("Downloaded file does not have a valid PDF signature")

        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"Downloaded {dest.name} ({size_mb:.1f} MB)")
        return True

    except Exception as exc:
        logger.warning(f"Failed to download {url}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def main(out_dir: str, skip_existing: bool = True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = DATASETS + FALLBACK_DATASETS
    downloaded = 0

    for item in targets:
        dest = out_dir / item["filename"]

        if skip_existing and _is_pdf_file(dest):
            logger.info(f"Skipping {item['filename']} (already exists and is a valid PDF)")
            downloaded += 1
            continue

        if dest.exists():
            logger.warning(f"Replacing invalid or partial file: {dest.name}")
            dest.unlink()

        logger.info(f"Resolving source for: {item['name']}")
        try:
            url = resolve_pdf_url(item)
        except Exception as exc:
            logger.error(f"Could not resolve a download URL for {item['name']}: {exc}")
            continue

        logger.info(f"Downloading: {item['name']}")
        success = download_file(url, dest)
        if success:
            downloaded += 1
        else:
            logger.error(f"Could not fetch: {item['name']}")
            logger.info(
                "Tip: open the handle page in a browser, download the PDF manually, "
                "place it in data/, then re-run index_documents.py."
            )

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Done. {downloaded}/{len(targets)} file(s) ready in {out_dir}/")

    pdfs = [p for p in out_dir.glob("*.pdf") if _is_pdf_file(p)]
    if pdfs:
        logger.info("Available PDFs:")
        for pdf in sorted(pdfs):
            size_mb = pdf.stat().st_size / (1024 * 1024)
            logger.info(f"  {pdf.name} ({size_mb:.1f} MB)")
    else:
        logger.warning("No valid PDFs found. Please download them manually.")
        sys.exit(1)

    invalid_pdfs = [p for p in out_dir.glob("*.pdf") if not _is_pdf_file(p)]
    if invalid_pdfs:
        logger.warning("Ignoring invalid PDF-like files:")
        for pdf in invalid_pdfs:
            logger.warning(f"  {pdf.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WHO health statistics PDFs")
    parser.add_argument("--out_dir", default="data/", help="Output directory for PDFs")
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Re-download even if the local files already look valid",
    )
    args = parser.parse_args()
    main(out_dir=args.out_dir, skip_existing=not args.no_skip)
