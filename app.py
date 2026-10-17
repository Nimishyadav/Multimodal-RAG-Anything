import os
import pickle
import json
import hashlib
import re

import streamlit as st
import streamlit.components.v1 as components

from src.document_reader import DOCUMENT_EXTENSIONS
from src.knowledge_graph import (
    save_graph, load_graph, query_graph,
    format_graph_answer
)
from src.ingest_pipeline import run_ingestion, rebuild_knowledge_graph
from src.pyvis_graph import render_graph

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs("storage", exist_ok=True)

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("MinerU Multimodal RAG")


from src.qa import answer_question


allowed_file_types = sorted({extension[1:] for extension in DOCUMENT_EXTENSIONS})

with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF, image, Excel, Word, or PowerPoint files",
        type=allowed_file_types,
        accept_multiple_files=True,
    )
    if uploaded_files:
        pending_uploads = st.session_state.setdefault("pending_uploads", set())
        uploaded_file_hashes = st.session_state.setdefault(
            "uploaded_file_hashes", {}
        )
        saved_upload_count = 0
        for uploaded in uploaded_files:
            filename = os.path.basename(uploaded.name.replace("\\", "/"))
            filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).rstrip(" .")
            if not filename or filename in {".", ".."}:
                st.error(f"Could not save upload with invalid filename: {uploaded.name!r}")
                continue

            file_stem = os.path.splitext(filename)[0].upper()
            if file_stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(
                r"(COM|LPT)[1-9]", file_stem
            ):
                filename = f"_{filename}"

            target_path = os.path.join(DATA_FOLDER, filename)
            uploaded_buffer = uploaded.getbuffer()
            content_hash = hashlib.sha256(uploaded_buffer).hexdigest()
            if uploaded_file_hashes.get(filename) == content_hash:
                continue

            try:
                with open(target_path, "wb") as file:
                    file.write(uploaded_buffer)
            except OSError as error:
                st.error(f"Could not save {filename!r} to data/: {error}")
                continue

            uploaded_file_hashes[filename] = content_hash
            saved_upload_count += 1
            pending_uploads.add(filename)
            st.session_state["knowledge_graph_source_choice"] = filename
        if saved_upload_count:
            st.success(f"Saved {saved_upload_count} file(s) to data/")
            st.info(
                "Click 'Process New Files (Ingest)' to update the knowledge graph "
                "for uploaded or replaced files."
            )

    st.divider()
    st.subheader("Files in data/")
    current_files = [f for f in sorted(os.listdir(DATA_FOLDER)) if not f.startswith("~$")]
    if current_files:
        for f in current_files:
            st.write(f"- {f}")
    else:
        st.write("_No files yet_")

    st.divider()
    if st.button("Process New Files (Ingest)", type="primary", use_container_width=True):
        status_box = st.empty()
        with st.spinner("Processing..."):
            pending_uploads = sorted(st.session_state.get("pending_uploads", set()))
            result = run_ingestion(
                file_names=pending_uploads or None,
                progress_callback=lambda msg: status_box.info(msg),
                force_reprocess=bool(pending_uploads),
            )
            if result.startswith("Done") or result.startswith("These files"):
                st.session_state["pending_uploads"] = set()
        st.success(result)

    image_files = [
        filename for filename in current_files
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"))
    ]
    if st.button(
        "Reprocess Image Files",
        use_container_width=True,
        disabled=not image_files,
    ):
        status_box = st.empty()
        with st.spinner("Reprocessing images and replacing their old graph data..."):
            result = run_ingestion(
                file_names=image_files,
                progress_callback=lambda msg: status_box.info(msg),
                force_reprocess=True,
            )
        st.success(result)

    if st.button("Rebuild Knowledge Graph", use_container_width=True):
        status_box = st.empty()
        with st.spinner("Rebuilding graph..."):
            result = rebuild_knowledge_graph(
                progress_callback=lambda msg: status_box.info(msg),
            )
        st.success(result)


tab_chat, tab_graph, tab_report, tab_history = st.tabs(
    ["Chat", "Knowledge Graph", "Readable Report", "Answer History"]
)


with tab_chat:
    st.subheader("Ask a question")

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input("Your question")
        submitted = st.form_submit_button("Ask")

    if submitted and query.strip():
        if not os.path.exists("storage/index.faiss"):
            st.warning("No index found yet. Upload files and click 'Process Files (Ingest)' first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, sources_used, debug = answer_question(query.strip())
                except Exception as e:
                    answer, sources_used, debug = f"Error: {e}", [], None
            st.session_state.chat_log.insert(0, (query.strip(), answer, sources_used, debug))

    for entry in st.session_state.chat_log:
        q, a, s = entry[:3]
        debug = entry[3] if len(entry) > 3 else None
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if s:
                st.caption("Sources: " + ", ".join(s))
            if debug:
                with st.expander("Retrieval debug details"):
                    metric_one, metric_two = st.columns(2)
                    metric_one.metric("Retrieval time", f"{debug['retrieval_time_ms']} ms")
                    metric_two.metric("Chunks used", debug["chunks_used"])
                    st.markdown("**Retrieved chunks**")
                    if debug["retrieved_chunks"]:
                        st.dataframe(debug["retrieved_chunks"], use_container_width=True)
                    else:
                        st.caption("No retrieved chunks for this response.")
                    st.markdown("**Graph triples**")
                    if debug["graph_triples"]:
                        st.dataframe(debug["graph_triples"], use_container_width=True)
                    else:
                        st.caption("No matching graph triples.")


# Exact Dropdown Filtering (All sources vs Specific File) with Compact Nodes
with tab_graph:
    st.subheader("Knowledge Graph Visualization")

    if not os.path.exists("storage/graph.pkl"):
        st.info("No graph yet. Upload files and click 'Process Files (Ingest)' first.")
    else:
        triples = load_graph()

        if not triples:
            st.warning("No entity relationships found in the knowledge graph.")
        else:
            # Extract list of all unique file sources from the graph
            all_sources = sorted(
                list(set(t.get("source", "") for t in triples if t.get("source")))
                + [f for f in os.listdir(DATA_FOLDER) if not f.startswith("~$")]
            )
            all_sources = sorted(list(set(all_sources)))

            graph_choice = st.selectbox(
                "Choose Graph View Source",
                ["All sources"] + all_sources,
                key="knowledge_graph_source_choice",
            )

            # Separate operational container triples from domain triples
            if graph_choice == "All sources":
                display_triples = [
                    t for t in triples
                    if str(t.get("relation", "")).strip().lower() not in ["contains_document", "has_graph_component"]
                ]
                if not display_triples:
                    display_triples = triples
                st.write(f"Showing **All Sources Combined Graph** ({len(display_triples)} relationships)")
            else:
                # Filter strictly for the selected file
                selected_triples = [
                    t for t in triples if t.get("source") == graph_choice
                ]
                # Filter out container metadata if domain triples exist
                clean_file_triples = [
                    t for t in selected_triples
                    if str(t.get("relation", "")).strip().lower() not in ["contains_document", "has_graph_component"]
                ]
                display_triples = clean_file_triples if clean_file_triples else selected_triples
                st.write(f"Showing Graph for **{graph_choice}** ({len(display_triples)} relationships)")

            if display_triples:
                graph_path = render_graph(
                    display_triples,
                    "storage/streamlit_graph.html",
                    merge_shared_entities=(graph_choice == "All sources"),
                )

                with open(graph_path, "r", encoding="utf-8") as file:
                    components.html(
                        file.read(),
                        height=850,
                        scrolling=True,
                    )
            else:
                st.warning(f"No relationships found for {graph_choice}. Try running 'Rebuild Knowledge Graph'.")

        st.divider()
        st.subheader("File Details")
        all_sources = sorted(
            set(t.get("source", "") for t in triples if t.get("source"))
            | {filename for filename in os.listdir(DATA_FOLDER) if not filename.startswith("~$")}
        )
        detail_choice = st.selectbox(
            "Select a file to read its complete extracted data",
            ["Select a file"] + all_sources,
        )
        if detail_choice != "Select a file":
            stats = {}
            source_text = ""
            try:
                with open("storage/stats.pkl", "rb") as file:
                    stats = pickle.load(file).get(detail_choice, {})
            except (FileNotFoundError, OSError, pickle.PickleError):
                pass
            try:
                with open("storage/source_text.pkl", "rb") as file:
                    source_text = pickle.load(file).get(detail_choice, "")
            except (FileNotFoundError, OSError, pickle.PickleError):
                pass

            if stats:
                st.json(stats)
            if source_text:
                st.text_area(
                    "Complete extracted file data",
                    source_text,
                    height=500,
                    key=f"details_{detail_choice}",
                )
            else:
                st.info("No extracted text is available. Run Process Files (Ingest) after adding the file.")


with tab_report:
    st.subheader("Readable Knowledge Graph Report")
    st.caption("This report updates whenever the graph source filter changes.")
    report_path = "storage/streamlit_selected_graph_readable.txt"
    if not os.path.exists(report_path):
        report_path = "storage/graph_readable.txt"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button("Download report", content, file_name="selected_graph_readable.txt")
        st.text_area("Report contents", content, height=600)
    else:
        st.info("No report yet. Run 'Process Files (Ingest)' first.")


with tab_history:
    st.subheader("Answer History")
    st.caption("This shows only the latest question and answer. The full archive remains in storage/all_files_answer_history.txt.")
    history_path = "storage/current_answer.txt"
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button("Download history", content, file_name="answer_history.txt")
        st.text_area("History contents", content, height=600)
    else:
        st.info("No answers saved yet.")
