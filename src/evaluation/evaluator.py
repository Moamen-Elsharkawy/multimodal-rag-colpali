"""
evaluation/evaluator.py

A lightweight evaluation harness that runs a predefined benchmark of queries
against the indexed documents and reports:

  - Retrieval accuracy  (did the correct page appear in top-k?)
  - Answer faithfulness (does the answer mention key expected terms?)
  - Average retrieval score
  - Per-modality breakdown: text-heavy, table, chart, mixed queries

The benchmark queries are tuned for WHO World Health Statistics reports.
If you swap the dataset, update BENCHMARK_QUERIES accordingly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from src.ingestion.indexer import DocumentIndex
from src.ingestion.embedder import ColPaliEmbedder
from src.retrieval.retriever import ColPaliRetriever
from src.retrieval.text_retriever import TextPageIndex, TextPageRetriever
from src.generation.generator import AnswerGenerator

load_dotenv()


# ---------------------------------------------------------------------------
# Benchmark queries — one per modality type
# Adjust these to match your actual PDF content after indexing.
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    {
        "id": "text_01",
        "query": "What is the global average life expectancy at birth?",
        "modality": "text",
        "expected_keywords": ["life expectancy", "years", "global"],
    },
    {
        "id": "table_01",
        "query": "Which country has the highest under-5 mortality rate?",
        "modality": "table",
        "expected_keywords": ["mortality", "rate", "country"],
    },
    {
        "id": "chart_01",
        "query": "What trend does the chart show for maternal mortality ratio between 2000 and 2020?",
        "modality": "chart",
        "expected_keywords": ["maternal", "mortality", "trend", "decrease"],
    },
    {
        "id": "mixed_01",
        "query": "What percentage of countries have achieved universal health coverage?",
        "modality": "mixed",
        "expected_keywords": ["universal", "health", "coverage", "percent"],
    },
    {
        "id": "text_02",
        "query": "What are the leading causes of death in low-income countries?",
        "modality": "text",
        "expected_keywords": ["pneumonia", "malaria", "diarrhea", "tuberculosis"],
    },
    {
        "id": "table_02",
        "query": "What is the neonatal mortality rate for Sub-Saharan Africa?",
        "modality": "table",
        "expected_keywords": ["neonatal", "Sub-Saharan", "Africa", "per 1000"],
    },
    {
        "id": "chart_02",
        "query": "Describe the regional distribution of HIV/AIDS prevalence shown in the figure.",
        "modality": "chart",
        "expected_keywords": ["HIV", "AIDS", "Africa", "prevalence"],
    },
    {
        "id": "mixed_02",
        "query": "How many people lack access to safely managed drinking water?",
        "modality": "mixed",
        "expected_keywords": ["water", "billion", "access", "safely managed"],
    },
]


@dataclass
class QueryResult:
    query_id: str
    query: str
    modality: str
    top_1_citation: str
    top_k_citations: List[str]
    retrieval_score: float
    answer: str
    keyword_hits: int
    keyword_total: int
    faithfulness_score: float        # keyword_hits / keyword_total
    latency_seconds: float


@dataclass
class EvalReport:
    total_queries: int
    avg_faithfulness: float
    avg_retrieval_score: float
    avg_latency: float
    modality_breakdown: Dict[str, float]  # modality → avg faithfulness
    results: List[QueryResult] = field(default_factory=list)


class RAGEvaluator:
    """
    Runs the benchmark suite and produces a structured report.

    Args:
        index_dir:  path to the saved FAISS index directory
        top_k:      number of pages to retrieve per query
        use_local:  pass True to use local Qwen2-VL instead of GPT-4o
    """

    def __init__(
        self,
        index_dir: str,
        top_k: int = 3,
        use_local: bool = False,
        model_name: str = "vidore/colpali-v1.3",
    ):
        logger.info("Loading index ...")
        self.embedder = None

        if TextPageIndex.can_load(index_dir):
            self.index = TextPageIndex.load(index_dir)
            self.retriever = TextPageRetriever(index=self.index, top_k=top_k)
            logger.info("Using TF-IDF fallback retriever for evaluation")
        else:
            self.index = DocumentIndex.load(index_dir)
            self.embedder = ColPaliEmbedder(model_name=model_name)
            self.retriever = ColPaliRetriever(
                index=self.index,
                embedder=self.embedder,
                top_k=top_k,
            )
            logger.info("Using ColPali retriever for evaluation")

        self.generator = AnswerGenerator(use_local=use_local)
        self.top_k = top_k

    def run(
        self,
        queries: Optional[List[dict]] = None,
        output_path: Optional[str] = None,
    ) -> EvalReport:
        """
        Execute all benchmark queries and return an EvalReport.

        Args:
            queries:     override the default BENCHMARK_QUERIES list
            output_path: if provided, save the report as JSON here
        """
        queries = queries or BENCHMARK_QUERIES
        results: List[QueryResult] = []

        for bq in queries:
            logger.info(f"Evaluating [{bq['id']}]: {bq['query'][:60]}")
            t0 = time.time()

            retrieved = self.retriever.retrieve(bq["query"])

            # load images for generation
            images, citations = [], []
            for r in retrieved:
                try:
                    images.append(Image.open(r.image_path).convert("RGB"))
                    citations.append(r.citation)
                except Exception as e:
                    logger.warning(f"Could not load image for page {r.page_number}: {e}")

            answer = "N/A"
            if images:
                answer = self.generator.generate(
                    question=bq["query"],
                    retrieved_images=images,
                    page_citations=citations,
                )

            latency = time.time() - t0

            # faithfulness = fraction of expected keywords in the answer
            answer_lower = answer.lower()
            expected = bq.get("expected_keywords", [])
            hits = sum(1 for kw in expected if kw.lower() in answer_lower)
            faith = hits / len(expected) if expected else 1.0

            results.append(QueryResult(
                query_id=bq["id"],
                query=bq["query"],
                modality=bq["modality"],
                top_1_citation=citations[0] if citations else "N/A",
                top_k_citations=citations,
                retrieval_score=retrieved[0].score if retrieved else 0.0,
                answer=answer,
                keyword_hits=hits,
                keyword_total=len(expected),
                faithfulness_score=faith,
                latency_seconds=round(latency, 2),
            ))

        # aggregate metrics
        avg_faith = sum(r.faithfulness_score for r in results) / len(results)
        avg_score = sum(r.retrieval_score for r in results) / len(results)
        avg_lat   = sum(r.latency_seconds for r in results) / len(results)

        modality_scores: Dict[str, List[float]] = {}
        for r in results:
            modality_scores.setdefault(r.modality, []).append(r.faithfulness_score)
        modality_breakdown = {m: sum(v) / len(v) for m, v in modality_scores.items()}

        report = EvalReport(
            total_queries=len(results),
            avg_faithfulness=round(avg_faith, 3),
            avg_retrieval_score=round(avg_score, 3),
            avg_latency=round(avg_lat, 2),
            modality_breakdown={k: round(v, 3) for k, v in modality_breakdown.items()},
            results=results,
        )

        self._print_report(report)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(asdict(report), f, indent=2)
            logger.info(f"Evaluation report saved to {output_path}")

        return report

    @staticmethod
    def _print_report(report: EvalReport):
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print("\n[bold green]=== Evaluation Report ===[/bold green]")
        console.print(f"Total queries   : {report.total_queries}")
        console.print(f"Avg faithfulness: {report.avg_faithfulness:.1%}")
        console.print(f"Avg retr. score : {report.avg_retrieval_score:.3f}")
        console.print(f"Avg latency     : {report.avg_latency:.2f}s")

        table = Table(title="Per-modality faithfulness")
        table.add_column("Modality")
        table.add_column("Score")
        for m, v in report.modality_breakdown.items():
            table.add_row(m, f"{v:.1%}")
        console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite")
    parser.add_argument("--index_dir", default="index/", help="Path to saved FAISS index")
    parser.add_argument("--output",    default="eval_results.json", help="Output JSON path")
    parser.add_argument("--top_k",     type=int, default=3)
    parser.add_argument("--local",     action="store_true", help="Use local VLM instead of GPT-4o")
    args = parser.parse_args()

    evaluator = RAGEvaluator(index_dir=args.index_dir, top_k=args.top_k, use_local=args.local)
    evaluator.run(output_path=args.output)
