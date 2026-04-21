# Technical Report - Multi-Modal RAG with ColPali

**Course:** DSAI 413 - Assignment 1  
**System:** Multi-Modal Document Intelligence (RAG-Based QA System)

## 1. Problem Setting

The assignment asks for a retrieval-augmented QA system that works on real documents containing text, tables, charts, figures, and layout-heavy pages. A text-only pipeline would require OCR, table extraction, and layout parsing before retrieval, and each of those stages can lose signal. This project instead starts from ColPali's core idea: treat each PDF page as an image and retrieve directly in visual space.

## 2. Architecture

The pipeline has four stages:

1. **Ingestion**
   - PDFs are rendered page-by-page with PyMuPDF.
   - Each page becomes an RGB image, preserving text, layout, charts, tables, and visual annotations.

2. **Embedding**
   - A ColPali-compatible retriever produces multi-vector page embeddings.
   - Each page is represented by many patch-level vectors rather than one dense vector, which keeps structural detail available at retrieval time.

3. **Retrieval**
   - Patch embeddings are stored in a FAISS inner-product index.
   - At query time, the query is embedded with the same model.
   - FAISS provides a fast candidate set, then MaxSim late interaction re-ranks candidate pages.

4. **Generation**
   - The top retrieved page images are sent to a multimodal generator.
   - The generator answers only from the retrieved context and cites the relevant page.
   - The implementation supports OpenAI by default and OpenRouter as a drop-in fallback when OpenAI is unavailable.

## 3. Dataset Choice

To avoid the same finance-heavy examples used in many public ColPali demos, this repo uses **WHO World Health Statistics 2022, 2023, and 2024**. This dataset is a good fit because it includes:

- dense country-level statistical tables
- charts tracking SDG indicators over time
- mixed text-and-figure layouts
- visually rich pages where layout matters for interpretation

The downloader resolves each file from the official WHO IRIS handle page and verifies that the saved file is a real PDF. This prevents stale or redirected HTML pages from contaminating the dataset.

## 4. Design Decisions

- **Page-as-image ingestion instead of OCR-first chunking**
  This matches ColPali's intended usage and keeps tables, chart structure, and layout intact.

- **Late interaction retrieval**
  ColPali-style retrieval is stronger than single-vector retrieval for visually complex pages because each query token can match a different region of the page.

- **FAISS for local experimentation**
  FAISS is simple, fast, and easy to demonstrate in an assignment-scale repo. It is appropriate for a few reports and keeps the system easy to explain.

- **OpenRouter fallback**
  API availability is often a practical constraint. Supporting OpenRouter through the same OpenAI-compatible interface makes the app easier to run on student machines when OpenAI access fails.

## 5. Evaluation Plan

The repo includes an evaluation harness with benchmark prompts across multiple modality types:

- text-heavy questions
- table-oriented questions
- chart interpretation questions
- mixed-layout questions

The evaluator reports:

- top retrieval citation
- retrieved page scores
- answer text
- keyword-based faithfulness proxy
- latency per query

For the final submission, the recommended workflow is:

1. build the WHO index
2. run the evaluator
3. record the output JSON
4. summarize the observed strengths and failure cases in the final report

## 6. Observations and Limitations

- The approach is strong for charts, tables, and mixed visual layouts because no manual chunking is required.
- The main practical bottleneck is model availability: ColPali needs either live Hugging Face access or a pre-downloaded local checkpoint.
- The current evaluator uses keyword overlap as a lightweight faithfulness signal; a stronger version would add human judging or an LLM-as-judge rubric.
- Cross-document comparison is still limited because retrieval is page-centric rather than explicitly multi-hop.

## 7. Conclusion

This project is a compact but realistic multi-modal RAG prototype. It demonstrates the full document-intelligence loop required by the assignment: ingestion, multi-modal retrieval, grounded answer generation, citations, and evaluation. The main strength of the system is that it keeps the original visual structure of the document available all the way from indexing to answer generation.

## References

1. Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models*, arXiv:2407.01449.
2. illuin-tech/colpali GitHub repository: https://github.com/illuin-tech/colpali
3. WHO IRIS repository: https://iris.who.int/
