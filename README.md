# Multi-Modal RAG System with ColPali

**DSAI 413 - Assignment 1**

A document intelligence system that answers questions over visually rich PDF reports by retrieving full page images instead of relying on brittle OCR-first chunking.

## Why ColPali

Traditional RAG pipelines flatten PDFs into text and lose the layout cues that matter most in tables, charts, maps, and mixed-format pages. ColPali takes a different approach:

- Each PDF page is rendered as an image.
- A vision-language retriever embeds the page into a multi-vector representation.
- Retrieval happens in visual space, so layout and non-text elements remain available.
- A multimodal generator answers from the retrieved pages and cites the source page.

This repo uses the WHO World Health Statistics reports as a deliberately different dataset from the usual finance-heavy ColPali demos.

## Architecture

```text
PDF Pages
    |
    v
[PDF -> Images]          src/ingestion/pdf_processor.py
    |
    v
[ColPali Embeddings]     src/ingestion/embedder.py
    |
    v
[FAISS Index]            src/ingestion/indexer.py
    |
    v
[MaxSim Retrieval]       src/retrieval/retriever.py
    |
    v
[Multimodal Generator]   src/generation/generator.py
    |
    v
[Streamlit App]          app/streamlit_app.py
```

## Dataset

The default dataset is:

- WHO World Health Statistics 2022
- WHO World Health Statistics 2023
- WHO World Health Statistics 2024

The downloader resolves the current PDF asset from the official WHO IRIS handle page, then validates that the downloaded file is a real PDF before keeping it.

## Project Structure

```text
multimodal-rag-colpali/
|-- src/
|   |-- ingestion/
|   |   |-- pdf_processor.py
|   |   |-- embedder.py
|   |   `-- indexer.py
|   |-- retrieval/
|   |   `-- retriever.py
|   |-- generation/
|   |   `-- generator.py
|   `-- evaluation/
|       `-- evaluator.py
|-- app/
|   `-- streamlit_app.py
|-- scripts/
|   |-- download_dataset.py
|   `-- index_documents.py
|-- data/
|-- index/
|-- report/
|   `-- technical_report.md
|-- requirements.txt
`-- .env.example
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env`, then choose one generation path:

```bash
# Option 1: OpenAI
OPENAI_API_KEY=...

# Option 2: OpenRouter (use this if OpenAI does not work on your machine)
GENERATION_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Notes:

- Do not commit real API keys.
- If you only want to build the dataset and index first, you can leave the generator keys empty and add them later.

### 3. Download the dataset

```bash
python scripts/download_dataset.py
```

If a previous run saved HTML files with a `.pdf` extension, the script will replace them automatically.

### 4. Build the index

```bash
python scripts/index_documents.py --data_dir data/ --index_dir index/
```

If Hugging Face access is blocked on your machine, download the ColPali or ColQwen checkpoint manually and pass the local path:

```bash
python scripts/index_documents.py --model C:/models/vidore-colpali-v1.3
```

If ColPali cannot be downloaded at all, the indexer now falls back to a local TF-IDF page retriever so the rest of the project can still run end to end with the extracted PDF text layer.

### 5. Launch the app

```bash
streamlit run app/streamlit_app.py
```

Inside the app:

- Use `Auto (OpenAI/OpenRouter)` for hosted multimodal generation.
- If OpenAI fails, configure `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.
- On this machine, OpenRouter is the working remote generation path.
- Use a local model path in the retriever field if Hugging Face downloads are blocked.

## Evaluation

Run the benchmark harness with:

```bash
python -m src.evaluation.evaluator --index_dir index/ --output eval_results.json
```

## Practical Notes

- ColPali-style retrieval preserves charts, tables, and page layout better than text-only chunking.
- FAISS is used here as a lightweight local store for page patch embeddings.
- The app supports OpenAI and OpenRouter through the same OpenAI-compatible client path.
- If OpenAI does not work for you, use OpenRouter.

## References

- ColPali GitHub: [https://github.com/illuin-tech/colpali](https://github.com/illuin-tech/colpali)
- ColPali paper: [https://arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449)
- WHO World Health Statistics collection: [https://iris.who.int/](https://iris.who.int/)
