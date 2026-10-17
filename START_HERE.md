# Mentor review package

MinerU parses supported documents first. Standalone images can receive local
Ollama vision enrichment. Sentence Transformers and FAISS provide vector
retrieval; the knowledge graph provides relationship evidence. CLI and Streamlit
use the same ingestion and grounded Q&A modules.

## Review without installation

Open storage/graph_selector.html in a browser to switch between All Sources and
the three source graphs. Keep the extracted folder structure intact.
Read storage/graph_readable.txt or storage/graph.json for the extracted results.
The sample data contains three flowchart images and 25 visible relationships.

## Run the application

Use a standard Windows Python 3.13 interpreter, then:

    python -m pip install -r requirements.txt
    ollama pull qwen3:4b
    ollama pull qwen2.5vl:3b
    python -m streamlit run app.py

Ollama must be running for answers. The sample FAISS index and graph are included.
For new ingestion, start the local MinerU server (mineru server start), then run:

    python ingest.py
    python visualize_graph.py --no-browser

Read README.md and .env.example for configuration and additional commands.
No credentials are included. Create your own .env if customization is needed.

## Validation and current limitation

23 regression tests passed. Real ingestion of all three sample images and HTML
graph generation completed successfully. Flowchart extraction still misses or
misidentifies some arrows; results need visual review for accuracy.
The optional tests/live_smoke.py is a separate live test and was not executed.

    python -m unittest discover -s tests -v

## Package contents

src/ and root scripts: implementation
tests/: regression tests and optional live smoke test
data/: the three example flowchart images
storage/: current graph, vector index, evidence and linked HTML pages
README.md, requirements.txt, .env.example: installation and configuration

Personal histories, .env, local models, environments, caches and old backups
are excluded.
