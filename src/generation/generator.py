"""
generation/generator.py

Takes a user question + a list of retrieved page images and generates a
grounded, citation-backed answer using GPT-4o Vision.

The prompt engineering here is intentional:
  - We send the page images as base64 along with the question
  - We explicitly ask the model to cite which page each claim comes from
  - We instruct it to say "I don't know" rather than hallucinate when
    the retrieved pages don't contain the answer

For a fully local setup (no API key), set use_local=True and the generator
will use Qwen2-VL-2B-Instruct instead — requires ~8GB VRAM.
"""

from __future__ import annotations

import base64
import io
import os
from typing import List, Optional

from PIL import Image
from loguru import logger


SYSTEM_PROMPT = """You are a precise document analyst. You will be given one or more pages \
from a document (as images) along with a user question. Your job is to:

1. Answer the question using ONLY the information visible in the provided pages.
2. Cite the specific page number(s) that support each claim (e.g., "According to page 5, ...").
3. If a question asks about a chart or figure, describe what the visual shows.
4. If the answer is not clearly present in the provided pages, say: "I couldn't find a clear \
answer in the retrieved pages."

Be concise and factual. Do not speculate beyond what the pages show."""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _image_to_base64(img: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class AnswerGenerator:
    """
    Generates answers grounded in retrieved page images.

    Args:
        model:       OpenAI model name, e.g. "gpt-4o"
        max_tokens:  maximum tokens in the response
        use_local:   if True, use a local Qwen2-VL model instead of GPT-4o
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 600,
        use_local: bool = False,
        provider: str = "auto",
    ):
        self.max_tokens = max_tokens
        self.use_local = use_local
        self.provider = provider

        if use_local:
            self.provider = "local"
            self.model = model or "Qwen/Qwen2-VL-2B-Instruct"
            self._init_local_model()
        else:
            self.provider, self.model = self._resolve_remote_provider(provider, model)
            self._init_remote_client()

    def _resolve_remote_provider(
        self,
        provider: str,
        model: Optional[str],
    ) -> tuple[str, str]:
        provider = provider.lower().strip()
        preferred_provider = os.getenv("GENERATION_PROVIDER", "").strip().lower()

        if provider == "auto":
            if preferred_provider in {"openai", "openrouter"}:
                provider = preferred_provider
            else:
                provider = "auto"

        if provider == "auto":
            if os.getenv("OPENAI_API_KEY"):
                return "openai", model or os.getenv("OPENAI_MODEL", "gpt-4o")
            if os.getenv("OPENROUTER_API_KEY"):
                return "openrouter", model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            raise EnvironmentError(
                "No remote generation credentials were found. "
                "Set OPENAI_API_KEY, or use OPENROUTER_API_KEY and OPENROUTER_MODEL "
                "if OpenAI is unavailable."
            )

        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            return "openai", model or os.getenv("OPENAI_MODEL", "gpt-4o")

        if provider == "openrouter":
            if not os.getenv("OPENROUTER_API_KEY"):
                raise EnvironmentError("OPENROUTER_API_KEY is not set.")
            return "openrouter", model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

        raise ValueError(f"Unsupported generator provider: {provider}")

    def _init_remote_client(self):
        from openai import OpenAI

        if self.provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
        else:
            headers = {}
            if os.getenv("OPENROUTER_SITE_URL"):
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
            if os.getenv("OPENROUTER_APP_NAME"):
                headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME")

            self.client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                default_headers=headers or None,
            )

        logger.info(f"Generator initialized with provider={self.provider}, model={self.model}")

    def _init_local_model(self):
        """Load Qwen2-VL locally as a fallback."""
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Local generation requires `transformers` and `accelerate`. "
                "Install them or use OpenAI/OpenRouter instead."
            ) from exc

        import torch

        local_model_name = self.model
        logger.info(f"Loading local VLM: {local_model_name}")

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        try:
            self._local_model = Qwen2VLForConditionalGeneration.from_pretrained(
                local_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
            self._local_processor = AutoProcessor.from_pretrained(local_model_name)
            logger.info("Local VLM loaded")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the local VLM '{local_model_name}'. "
                "If the model is not cached locally, use OpenAI/OpenRouter or "
                "download the model manually."
            ) from exc

    def generate(
        self,
        question: str,
        retrieved_images: List[Image.Image],
        page_citations: List[str],
    ) -> str:
        """
        Generate a grounded answer from retrieved page images.

        Args:
            question:         the user's question
            retrieved_images: list of PIL images (the top-k retrieved pages)
            page_citations:   matching citation strings, e.g. ["Report 2023, page 4", ...]

        Returns:
            A string answer with inline citations
        """
        if self.use_local:
            return self._generate_local(question, retrieved_images, page_citations)
        return self._generate_remote(question, retrieved_images, page_citations)

    def _generate_remote(
        self,
        question: str,
        retrieved_images: List[Image.Image],
        page_citations: List[str],
    ) -> str:
        """Call GPT-4o with the images and question."""

        # build the content array: label each page then embed its image
        content = []

        for idx, (img, citation) in enumerate(zip(retrieved_images, page_citations)):
            content.append({
                "type": "text",
                "text": f"[Retrieved page {idx + 1}: {citation}]"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_image_to_base64(img)}",
                    "detail": "high",
                }
            })

        content.append({
            "type": "text",
            "text": f"\nQuestion: {question}"
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Answer generation failed via {self.provider} using model '{self.model}'. "
                "If OpenAI is unavailable on your machine, configure OpenRouter instead."
            ) from exc

        message = response.choices[0].message.content
        if isinstance(message, str):
            return message.strip()
        return str(message).strip()

    def _generate_local(
        self,
        question: str,
        retrieved_images: List[Image.Image],
        page_citations: List[str],
    ) -> str:
        """Use local Qwen2-VL as a fallback when no API key is available."""
        import torch

        # Qwen2-VL supports multiple images in a single prompt
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in retrieved_images],
                    {"type": "text", "text": SYSTEM_PROMPT + f"\n\nQuestion: {question}"},
                ],
            }
        ]

        text = self._local_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._local_processor(
            text=[text],
            images=retrieved_images,
            return_tensors="pt",
        ).to(self._local_model.device)

        with torch.no_grad():
            output_ids = self._local_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._local_processor.decode(generated[0], skip_special_tokens=True).strip()
