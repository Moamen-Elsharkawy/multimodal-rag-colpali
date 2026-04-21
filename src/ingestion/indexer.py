"""
ingestion/indexer.py

Manages the FAISS-based vector index and the metadata store.

The storage design is a bit unconventional for multi-vector search because
standard FAISS is built for single-vector lookup. Here's the approach:

  - Each page produces M patch embeddings (e.g. M ≈ 1030).
  - We flatten all patches across all pages into a single FAISS flat index.
  - We keep a separate "page_map" array that maps every flat index entry back
    to its originating page ID.
  - At retrieval time, we DON'T use FAISS for final scoring — we use it only
    for a fast approximate pre-filter (top-100 candidate patches), then rerank
    with full MaxSim scoring (see retriever.py).

This avoids having to store ragged tensors in FAISS while keeping lookups fast.

Metadata (doc_name, doc_path, page_number, image path) is pickled alongside
the FAISS index so the retriever can reconstruct full citations.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
from loguru import logger


@dataclass
class PageMeta:
    """Everything we need to build a citation and display a result."""
    page_id: int
    doc_name: str
    doc_path: str
    page_number: int         # 0-indexed
    image_path: str          # saved PNG on disk


class DocumentIndex:
    """
    Wraps a FAISS flat index and a metadata store for multi-vector page retrieval.

    Usage:
        index = DocumentIndex(embed_dim=128)
        index.add_pages(page_metas, page_embeddings)
        index.save("index/")

        index2 = DocumentIndex.load("index/")
        candidate_page_ids = index2.search_patches(query_emb, top_k_patches=200)
    """

    INDEX_FILE = "faiss_flat.index"
    META_FILE  = "page_metadata.pkl"
    EMBS_FILE  = "page_embeddings.pkl"

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.page_map: List[int] = []        # flat_idx → page_id
        self.page_metas: Dict[int, PageMeta] = {}
        self.page_embeddings: Dict[int, torch.Tensor] = {}  # page_id → (M, D)

    def add_pages(
        self,
        metas: List[PageMeta],
        embeddings: List[torch.Tensor],
    ) -> None:
        """
        Add a batch of pages to the index.

        Args:
            metas:       list of PageMeta, one per page
            embeddings:  list of tensors, each shaped (n_patches, embed_dim)
        """
        assert len(metas) == len(embeddings), "metas and embeddings must have the same length"

        all_patches: List[np.ndarray] = []

        for meta, emb in zip(metas, embeddings):
            page_id = meta.page_id
            self.page_metas[page_id] = meta
            self.page_embeddings[page_id] = emb

            # normalize patch vectors for inner-product search
            patches_np = emb.float().numpy()  # (M, D)
            patches_np = patches_np / (np.linalg.norm(patches_np, axis=1, keepdims=True) + 1e-9)

            # record which page each patch belongs to
            self.page_map.extend([page_id] * len(patches_np))
            all_patches.append(patches_np)

        patches_matrix = np.vstack(all_patches).astype(np.float32)

        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self.embed_dim)

        self.faiss_index.add(patches_matrix)
        logger.info(
            f"Index now has {self.faiss_index.ntotal} patch vectors "
            f"across {len(self.page_metas)} pages"
        )

    def save(self, index_dir: str) -> None:
        """Persist the index and metadata to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(index_dir / self.INDEX_FILE))

        meta_payload = {
            "page_map":  self.page_map,
            "page_metas": self.page_metas,
            "embed_dim": self.embed_dim,
        }
        with open(index_dir / self.META_FILE, "wb") as f:
            pickle.dump(meta_payload, f)

        with open(index_dir / self.EMBS_FILE, "wb") as f:
            pickle.dump(self.page_embeddings, f)

        logger.info(f"Index saved to {index_dir}")

    @classmethod
    def load(cls, index_dir: str) -> "DocumentIndex":
        """Load a previously saved index from disk."""
        index_dir = Path(index_dir)

        faiss_path = index_dir / cls.INDEX_FILE
        meta_path  = index_dir / cls.META_FILE
        embs_path  = index_dir / cls.EMBS_FILE

        for p in [faiss_path, meta_path, embs_path]:
            if not p.exists():
                raise FileNotFoundError(f"Index file missing: {p}")

        with open(meta_path, "rb") as f:
            meta_payload = pickle.load(f)

        with open(embs_path, "rb") as f:
            page_embeddings = pickle.load(f)

        idx = cls(embed_dim=meta_payload["embed_dim"])
        idx.faiss_index   = faiss.read_index(str(faiss_path))
        idx.page_map      = meta_payload["page_map"]
        idx.page_metas    = meta_payload["page_metas"]
        idx.page_embeddings = page_embeddings

        logger.info(
            f"Loaded index: {idx.faiss_index.ntotal} patches, "
            f"{len(idx.page_metas)} pages"
        )
        return idx

    def get_candidate_page_ids(
        self,
        query_patch_emb: np.ndarray,
        top_k_patches: int = 200,
    ) -> List[int]:
        """
        Fast approximate pre-filter: find the top_k_patches closest patches
        to the query embedding and return the unique page IDs they come from.

        The caller (retriever.py) then does exact MaxSim over these candidates.
        """
        # query_patch_emb: (query_len, D)  — we search with each query token
        query_patch_emb = query_patch_emb.astype(np.float32)
        query_patch_emb = query_patch_emb / (
            np.linalg.norm(query_patch_emb, axis=1, keepdims=True) + 1e-9
        )

        # search all query tokens at once
        _, indices = self.faiss_index.search(query_patch_emb, top_k_patches)

        # collect unique page IDs from the patch hits
        candidate_ids = set()
        for row in indices:
            for flat_idx in row:
                if 0 <= flat_idx < len(self.page_map):
                    candidate_ids.add(self.page_map[flat_idx])

        return list(candidate_ids)
