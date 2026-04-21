"""
retrieval/retriever.py

Implements ColBERT-style MaxSim scoring for multi-vector retrieval.

Given a query embedding Q (shape: query_len × D) and a page embedding P
(shape: n_patches × D), the score is:

    score(Q, P) = Σ_{i=1}^{query_len}  max_{j=1}^{n_patches}  Q[i] · P[j]

This "late interaction" formula lets every query token find its best-matching
visual patch independently — much richer than a single dot product.

Flow:
  1. Embed the query with ColPali
  2. Use FAISS pre-filter to get ~50–100 candidate pages
  3. Run exact MaxSim over those candidates
  4. Return top-k pages sorted by score, with metadata for citation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from loguru import logger

from src.ingestion.indexer import DocumentIndex, PageMeta
from src.ingestion.embedder import ColPaliEmbedder


@dataclass
class RetrievedPage:
    """A single retrieval result with score and citation info."""
    page_id: int
    score: float
    doc_name: str
    doc_path: str
    page_number: int       # 0-indexed (display as page_number + 1)
    image_path: str

    @property
    def citation(self) -> str:
        return f"{self.doc_name}, page {self.page_number + 1}"


class ColPaliRetriever:
    """
    End-to-end retriever: takes a text query, returns ranked PageRecord results.

    Args:
        index:    a loaded DocumentIndex
        embedder: a ColPaliEmbedder (shared with ingestion to avoid double-loading)
        top_k:    number of pages to return
        pre_filter_patches: how many FAISS patch hits to use for candidate selection
    """

    def __init__(
        self,
        index: DocumentIndex,
        embedder: ColPaliEmbedder,
        top_k: int = 3,
        pre_filter_patches: int = 300,
    ):
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.pre_filter_patches = pre_filter_patches

    def retrieve(self, query: str) -> List[RetrievedPage]:
        """
        Retrieve the top-k most relevant pages for a query.

        Args:
            query: natural language question

        Returns:
            list of RetrievedPage, sorted by descending score
        """
        # Step 1: embed the query
        query_emb = self.embedder.embed_query(query)   # (query_len, D)
        query_np  = query_emb.float().numpy()

        # Step 2: FAISS pre-filter → candidate page IDs
        candidate_ids = self.index.get_candidate_page_ids(
            query_np, top_k_patches=self.pre_filter_patches
        )

        if not candidate_ids:
            logger.warning("FAISS pre-filter returned no candidates — using all pages")
            candidate_ids = list(self.index.page_metas.keys())

        logger.debug(f"Pre-filter: {len(candidate_ids)} candidate pages for '{query[:60]}'")

        # Step 3: exact MaxSim scoring
        scored: List[tuple[float, int]] = []

        for page_id in candidate_ids:
            page_emb = self.index.page_embeddings[page_id]   # (M, D)
            score    = self._maxsim(query_emb, page_emb)
            scored.append((score, page_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = scored[:self.top_k]

        # Step 4: build result objects
        results: List[RetrievedPage] = []
        for score, page_id in top_results:
            meta: PageMeta = self.index.page_metas[page_id]
            results.append(RetrievedPage(
                page_id=page_id,
                score=float(score),
                doc_name=meta.doc_name,
                doc_path=meta.doc_path,
                page_number=meta.page_number,
                image_path=meta.image_path,
            ))

        return results

    @staticmethod
    def _maxsim(query_emb: torch.Tensor, page_emb: torch.Tensor) -> float:
        """
        Compute the MaxSim score between a query and a page.

        query_emb: (query_len, D)
        page_emb:  (n_patches, D)
        """
        # similarity matrix: (query_len, n_patches)
        sim = torch.matmul(query_emb.float(), page_emb.float().T)
        # max over patches for each query token, then sum
        return float(sim.max(dim=1).values.sum())
