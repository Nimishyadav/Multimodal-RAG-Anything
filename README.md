# Multimodal RAGAnything

A local multimodal RAGAnything system built around a MinerU-first
multimodal GraphRAG architecture for question answering across PDFs,
DOCX, PPTX, Excel workbooks, and standalone images.

MinerU provides the primary local parsing and ingestion layer. Local
Ollama models handle visual enrichment, relationship extraction, and
grounded question answering. Sentence Transformers and FAISS provide
semantic retrieval, while the Knowledge Graph and PyVis support
relationship-based retrieval and interactive graph visualization.

---

## Features

- Local MinerU parsing for:
  - PDF
  - DOCX
  - PPTX
  - XLSX
  - XLS
  - Standalone images

- Local Ollama models:
  - `qwen3:4b` for text generation and knowledge-graph extraction
  - `qwen2.5vl:3b` for image and flowchart understanding

- FAISS-based semantic vector retrieval

- Grounded question answering using retrieved document and graph evidence

- Knowledge Graph construction from explicit document relationships

- Multimodal image and flowchart understanding:
  - visible labels
  - nodes
  - node shapes
  - directed connectors
  - branch labels
  - structured flowchart transitions

- Interactive PyVis Knowledge Graph visualization

- Graph filtering:
  - All sources
  - Individual source files

- Hierarchical top-to-bottom layout for flowchart graphs

- Streamlit dashboard for document upload, ingestion, graph management,
  visualization, and question answering

- Command-line ingestion and chat interface

- Incremental ingestion:
  - New files are processed
  - Changed files are reprocessed
  - Unchanged files are reused

- Local processing with no paid API keys or remote parsing services

---

## Architecture

The overall workflow is:

```text
PDF / DOCX / PPTX / XLSX / XLS / Image
                    |
                    v
              MinerU Parser
                    |
                    v
        Structured Markdown / HTML
                    |
                    v
              Chunking Layer
                    |
          +---------+---------+
          |                   |
          v                   v
 Sentence Transformers    Relationship Extraction
          |                   |
          v                   v
       FAISS             Knowledge Graph
          |                   |
          +---------+---------+
                    |
                    v
             Graph + Vector
                Evidence
                    |
                    v
              Local Ollama
                    |
                    v
           Grounded Answer
                    |
                    v
             Streamlit / CLI
