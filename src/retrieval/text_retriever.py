"""
retrieval/text_retriever.py

Fallback text-based retrieval for environments where ColPali cannot be loaded.

This keeps the project runnable when Hugging Face access is blocked by using the
PDF text layer and a lightweight TF-IDF retriever. It is not a replacement for
ColPali, but it preserves the rest of the pipeline so indexing, retrieval,
generation, and the Streamlit demo still work end to end.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.ingestion.indexer import PageMeta
from src.retrieval.retriever import RetrievedPage


class TextPageIndex:
    """A persisted TF-IDF index over page-level extracted text."""

    INDEX_FILE = "text_page_index.pkl"

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.page_matrix = None
        self.page_metas: Dict[int, PageMeta] = {}
        self.page_texts: Dict[int, str] = {}

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip()
        if text:
            return text
        # Empty pages should still be retrievable weakly via metadata.
        return "empty page"

    def add_pages(
        self,
        metas: List[PageMeta],
        texts: List[str],
    ) -> None:
        assert len(metas) == len(texts), "metas and texts must have the same length"

        corpus = []
        for meta, text in zip(metas, texts):
            self.page_metas[meta.page_id] = meta
            normalized = self._normalize_text(text)
            self.page_texts[meta.page_id] = normalized
            corpus.append(normalized)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
        )
        self.page_matrix = self.vectorizer.fit_transform(corpus)
        logger.info(f"Built TF-IDF fallback index for {len(metas)} pages")

    def save(self, index_dir: str) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "vectorizer": self.vectorizer,
            "page_matrix": self.page_matrix,
            "page_metas": self.page_metas,
            "page_texts": self.page_texts,
        }
        with open(index_dir / self.INDEX_FILE, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"Text fallback index saved to {index_dir}")

    @classmethod
    def can_load(cls, index_dir: str) -> bool:
        return (Path(index_dir) / cls.INDEX_FILE).exists()

    @classmethod
    def load(cls, index_dir: str) -> "TextPageIndex":
        index_path = Path(index_dir) / cls.INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(f"Text index file missing: {index_path}")

        with open(index_path, "rb") as f:
            payload = pickle.load(f)

        idx = cls()
        idx.vectorizer = payload["vectorizer"]
        idx.page_matrix = payload["page_matrix"]
        idx.page_metas = payload["page_metas"]
        idx.page_texts = payload["page_texts"]
        logger.info(f"Loaded text fallback index with {len(idx.page_metas)} pages")
        return idx


class TextPageRetriever:
    """TF-IDF page retriever with the same output shape as the ColPali retriever."""

    def __init__(self, index: TextPageIndex, top_k: int = 3):
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedPage]:
        if self.index.vectorizer is None or self.index.page_matrix is None:
            raise RuntimeError("Text fallback index is not initialized")

        query_vec = self.index.vectorizer.transform([query])
        scores = linear_kernel(query_vec, self.index.page_matrix).ravel()

        ranked_ids = np.argsort(scores)[::-1][: self.top_k]
        page_ids = list(self.index.page_metas.keys())

        results: List[RetrievedPage] = []
        for idx in ranked_ids:
            page_id = page_ids[idx]
            meta = self.index.page_metas[page_id]
            results.append(
                RetrievedPage(
                    page_id=page_id,
                    score=float(scores[idx]),
                    doc_name=meta.doc_name,
                    doc_path=meta.doc_path,
                    page_number=meta.page_number,
                    image_path=meta.image_path,
                )
            )

        return results
