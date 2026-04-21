"""
ingestion/embedder.py

Wraps ColPali (or ColQwen2 as a drop-in) and produces multi-vector embeddings
for document pages and text queries.

ColPali's key idea: instead of encoding a document to a single dense vector,
it produces one embedding per image patch (typically ~1030 patches for a
448×448 image). At query time, the MaxSim operation scores every query token
against every patch and sums the max scores — this is the "late interaction"
from ColBERT, applied in visual space.

We support both ColPali v1.3 (Gemma-based) and ColQwen2 (Apache 2.0, slightly
stronger on the ViDoRe benchmark). Default is ColPali v1.3.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm

# lazy imports — only pulled in when the embedder is actually used
_colpali_loaded = False
_model = None
_processor = None
_model_class = None
_processor_class = None


def _load_model(model_name: str):
    """Load ColPali or ColQwen2 into memory (singleton pattern)."""
    global _colpali_loaded, _model, _processor, _model_class, _processor_class

    if _colpali_loaded:
        return

    from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor

    if "qwen" in model_name.lower():
        _model_class = ColQwen2
        _processor_class = ColQwen2Processor
    else:
        _model_class = ColPali
        _processor_class = ColPaliProcessor

    device = _pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    logger.info(f"Loading {model_name} on {device} ({dtype}) ...")

    try:
        _model = _model_class.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        ).eval()

        _processor = _processor_class.from_pretrained(model_name)
        _colpali_loaded = True
        logger.info("Model loaded successfully")
    except Exception as exc:
        _model = None
        _processor = None
        _colpali_loaded = False

        local_hint = (
            "Use a local checkpoint path if Hugging Face is blocked, for example "
            "`python scripts/index_documents.py --model C:/models/vidore-colpali-v1.3`."
        )
        if Path(model_name).exists():
            local_hint = "The provided local model path exists, but loading it still failed."

        raise RuntimeError(
            f"Failed to load the ColPali model '{model_name}'. "
            "This project needs either working access to huggingface.co or a "
            f"pre-downloaded local checkpoint. {local_hint}"
        ) from exc


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ColPaliEmbedder:
    """
    Produces multi-vector embeddings for pages and queries using ColPali.

    Args:
        model_name: HuggingFace model ID, e.g. "vidore/colpali-v1.3"
        batch_size:  number of images to embed in one forward pass. Keep
                     this at 2–4 if you're on a 8GB GPU with full precision.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        _load_model(model_name)

    @property
    def model(self):
        return _model

    @property
    def processor(self):
        return _processor

    def embed_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Embed a list of page images.

        Returns a list of tensors, each shaped (n_patches, embed_dim).
        n_patches is typically around 1030 for ColPali at 448px resolution.
        """
        all_embeddings: List[torch.Tensor] = []

        for i in tqdm(range(0, len(images), self.batch_size), desc="Embedding pages"):
            batch_imgs = images[i: i + self.batch_size]
            inputs = self.processor.process_images(batch_imgs).to(self.model.device)

            with torch.no_grad():
                batch_embs = self.model(**inputs)  # (batch, n_patches, dim)

            # unbind along batch dimension → list of (n_patches, dim) tensors
            for emb in torch.unbind(batch_embs):
                all_embeddings.append(emb.cpu())

        return all_embeddings

    def embed_query(self, query: str) -> torch.Tensor:
        """
        Embed a single text query.

        Returns a tensor shaped (query_len, embed_dim).
        """
        inputs = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            emb = self.model(**inputs)  # (1, query_len, dim)
        return emb[0].cpu()

    def embed_queries(self, queries: List[str]) -> List[torch.Tensor]:
        """Batch version of embed_query."""
        embeddings = []
        for q in queries:
            embeddings.append(self.embed_query(q))
        return embeddings
