# MinerU Multimodal RAG

Local, multimodal retrieval-augmented generation for PDFs, images, Excel
workbooks, and Word documents. MinerU parses every supported input locally;
Ollama provides local text answers, image understanding, and knowledge-graph
extraction. FAISS handles vector retrieval and PyVis renders the graph.

## Features

- Local MinerU parsing for PDFs, images, `.xlsx`, `.xls`, and `.docx`
- Local Ollama models (`qwen3:4b` and `qwen2.5vl:3b`) for text answers and
  image/flowchart analysis
- FAISS vector search with grounded question answering and summaries
- Knowledge-graph extraction and interactive graph visualization:
  a graph selector for the combined All sources view or an individual file,
  using a directed hierarchical layout for flowcharts and the existing layout
  for general document graphs
- Streamlit dashboard and command-line chat
- No paid API keys or remote parsing services

## Requirements

- Windows with standard 64-bit Python 3.13 (not MSYS2/UCRT64 Python)
- Ollama installed and running locally
- MinerU 4 and its local parsing models

Install dependencies and pull the local model:

```powershell
py -3.13 -m pip install -r requirements.txt
ollama pull qwen3:4b
ollama pull qwen2.5vl:3b
```

Start MinerU's local server in a terminal and leave it running:

```powershell
mineru server start
```

The project uses MinerU's local `flash` tier by default. It does not send
documents to MinerU's remote service. To allow more time for large documents,
copy `.env.example` to `.env` and set `MINERU_WAIT_SECONDS` (default: `300`).

In VS Code, select the same standard Python 3.13 interpreter for the workspace:
**Python: Select Interpreter** → the `Python313\python.exe` installation.
This ensures packages such as `ollama` are installed in the interpreter that
runs the project.

## Usage

Run the Streamlit dashboard:

```powershell
py -3.13 -m streamlit run app.py
```

The dashboard accepts uploads and can incrementally ingest new files. After
uploading a file or replacing one with the same name, click
**Process New Files (Ingest)** to update embeddings and its knowledge graph.
The command-line ingestion uses the same pipeline:

```powershell
py -3.13 ingest.py
py -3.13 ingest.py --reprocess-images
py -3.13 ingest.py --rebuild-graph
```

The default command uses the same ingestion pipeline as the dashboard. It ingests
new or changed supported files and reuses unchanged files. File hashes and parser/model
settings are saved with the source statistics; old storage is refreshed once when
these signatures are missing. `--reprocess-images`
re-reads every supported image with the configured local vision model and replaces its saved graph
data; `--rebuild-graph` rebuilds only from saved content. It cannot recover missing
visual relationships from an old image-link-only parse: reprocess images first.
The dashboard's
**Reprocess Image Files** button performs the same image refresh.
Image ingestion first parses the image with MinerU, then uses the local
`qwen2.5vl:3b` vision model to identify visible labels, node shapes, and
explicitly directed connectors. Flowchart graph edges are created only from
validated structured vision transitions; image text is never sent to the
generic text graph extractor and nearby nodes are never linked as a fallback.
Unlabeled or malformed transitions are omitted and reported. Branch labels are
preserved as edge details. Flowcharts use a top-to-bottom hierarchical directed
layout; node shape metadata is shown as start/end ellipses, decision diamonds,
and process boxes. General document graphs retain their existing layout.
Repeated same-label shapes in diagrams stay distinct by image and node ID.
Select one source in the graph filter to focus the view.

Existing `storage/graph.pkl` data can contain relationships extracted by older
logic. After changing the vision prompt or graph conversion, click
**Reprocess Image Files** in the Streamlit sidebar to re-read images with local
the configured local vision model and replace their saved image triples. This does not delete uploaded
files or unrelated document data. **Rebuild Knowledge Graph** only rebuilds
from already saved chunks; it does not re-read an image or refresh the vision
model output.

MinerU is the document parser; it does not provide RAG embeddings or generate
chat answers. The project uses SentenceTransformer for vector embeddings and
local Ollama models for image understanding, knowledge-graph extraction, and
chat answers. Ingestion now reports its current stage and periodically reports
while waiting for MinerU to finish parsing.

## Shared workflow and optional enrichment

PDF, DOCX, PPTX, XLS/XLSX and standalone images all enter through MinerU.
Text/table chunks feed the local text model for explicit relationships. Standalone
images can receive vision enrichment; observed arrows, branch labels and nodes are
saved with the image chunks for later graph rebuilds. MinerU page markers become
page metadata, and both Markdown and HTML tables retain their table modality.
FAISS and graph evidence are combined for grounded answers through `src/qa.py`,
shared by `chat.py` and the dashboard. Source-specific questions filter both evidence
paths. PyVis outputs include the combined graph and separate source pages.

Model/parsing work completes before newly generated storage is published. A parser,
model or embedding failure leaves the previous saved artifacts available. The file
replacements are sequential; this is not a database transaction against power loss
or concurrent ingestion. Run one ingestion at a time.

Settings in `.env` (see `.env.example`):

```dotenv
RAG_TEXT_MODEL=qwen3:4b
RAG_VISION_MODEL=qwen2.5vl:3b
RAG_VISUAL_ENRICHMENT=true
RAG_USE_RERANKER=false
```

Use the names installed in `ollama list`. `llama3` and `qwen3-vl:8b-instruct`
are configurable alternatives; the project defaults to Qwen3 4B for text and Qwen2.5-VL 3B for images.
Use an explicit Qwen3-VL instruct tag if choosing that family: generic Qwen3-VL
tags can emit reasoning without final content despite think=false.
This behavior is reported in the [Ollama issue tracker](https://github.com/ollama/ollama/issues/16945).
Vision inference limits the longest image side to 1024 pixels by default; set
`RAG_VISION_MAX_SIDE` higher for very small labels. This preparation never changes
the uploaded source image. Vision requests use streaming, with an early stop on
whitespace-only responses and retries with/without the JSON formatter.
Reranking is optional and disabled by default to avoid an additional model pass.
Enable it with `RAG_USE_RERANKER=true` when you need that pass.

```powershell
python ingest.py --source report.pdf
python ingest.py --force
python ingest.py --no-vision
python visualize_graph.py --no-browser
python chat.py "What is the relationship between the entities?"
```

`--no-vision` keeps MinerU as the parser and skips standalone-image enrichment.
Image references alone do not contain readable nodes/arrows, so this mode cannot
produce a flowchart graph from those references. Keep vision enabled for flowcharts.
Neither PDF nor Office documents are sent to Ollama vision for parsing.

Ollama generation has no configured output-token cap (`num_predict = -1`) and
has a 30-minute request timeout. Generation can therefore run for a long time
or use substantial local compute; the model's context capacity still applies.

## Project Structure

```text
src/
├── mineru_reader.py
├── image_reader.py
├── llm_client.py
├── config.py
├── qa.py
├── ingest_pipeline.py
├── knowledge_graph.py
├── retriever.py
├── pyvis_graph.py
├── embedder.py
├── chunker.py
└── vector_store.py

app.py
chat.py
ingest.py
visualize_graph.py
requirements.txt
```

Contract tests can run without model services:

```powershell
python -m unittest discover -s tests -v
```

An opt-in live smoke test uses a temporary PDF and temporary storage, requires
MinerU/Ollama services and a cached embedding model, and leaves project storage
unchanged:

```powershell
python tests/live_smoke.py
```
