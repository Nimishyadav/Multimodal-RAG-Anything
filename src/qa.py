import os
import json
import time
from datetime import datetime
from src.knowledge_graph import load_graph, query_graph, format_graph_answer
from src.retriever import (
    search, get_summary_chunks, is_summary_query, is_stats_query, get_stats_answer,
    is_graph_query, get_known_sources, find_mentioned_source, build_grounded_context,
    is_excel_relationship_query, get_excel_chunks, is_excel_query,
    asks_for_all_excel_files, build_excel_relationship_context,
    is_image_query, get_image_chunks,
)
from src.llm import generate_answer, generate_summary, classify_intent

def save_answer_record(query, answer, sources):
    """Same readable-history logic used by chat.py — every answer gets
    written to plain text/JSONL files under storage/, so anyone can open
    and read them without touching any code."""
    query_lower = query.lower()
    is_graph_record = any(
        term in query_lower
        for term in ("graph", "relationship", "relationships", "entity", "entities", "connected")
    )
    if any(term in query_lower for term in ("excel", "spreadsheet", "workbook", "worksheet", ".xlsx", ".xls")):
        category = "excel"
    elif any(term in query_lower for term in ("image", "picture", "photo", "diagram", ".png", ".jpg", ".jpeg")):
        category = "image"
    elif any(term in query_lower for term in ("ppt", "pptx", "presentation", "powerpoint", "slide")):
        category = "ppt"
    elif any(term in query_lower for term in ("pdf", "document")):
        category = "pdf"
    else:
        category = "mixed"

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "category": category,
        "query": query,
        "sources": sorted(set(sources)),
        "answer": answer,
    }
    os.makedirs("storage", exist_ok=True)

    with open("storage/answer_history.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open("storage/current_answer.json", "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)

    history_text = (
        f"[{record['timestamp']}] CATEGORY: {category}\n"
        f"QUESTION: {query}\n"
        f"SOURCES: {', '.join(record['sources']) or 'none'}\n"
        f"ANSWER:\n{answer}\n\n{'=' * 80}\n\n"
    )

    with open("storage/all_files_answer_history.txt", "a", encoding="utf-8") as file:
        file.write(history_text)
    with open("storage/current_answer.txt", "w", encoding="utf-8") as file:
        file.write(history_text)
    with open(f"storage/{category}_answer_history.txt", "a", encoding="utf-8") as file:
        file.write(history_text)
    if is_graph_record:
        with open("storage/graph_answer_history.txt", "a", encoding="utf-8") as file:
            file.write(history_text)


def answer_question(query):
    """Combine vector/graph evidence for both terminal and dashboard Q&A."""
    started_at = time.perf_counter()
    debug_chunks = []
    debug_triples = []

    if is_stats_query(query):
        intent = "STATS"
    elif is_graph_query(query):
        intent = "GRAPH"
    elif is_image_query(query) and not is_excel_query(query):
        intent = "SUMMARY"
    elif is_excel_query(query):
        intent = "GRAPH" if is_excel_relationship_query(query) else "SUMMARY"
    elif is_summary_query(query):
        intent = "SUMMARY"
    else:
        intent = classify_intent(query)

    sources_used = []

    if intent == "STATS":
        answer = get_stats_answer()

    elif intent == "GRAPH":
        sources = get_known_sources()
        target_source = find_mentioned_source(query, sources)
        triples = load_graph()

        is_overview_query = any(
            phrase in query.lower()
            for phrase in (
                "show the knowledge graph", "show knowledge graph", "view the graph",
                "all relationships", "all entities", "graph of everything",
                "relationships in all files", "knowledge graph of all",
                "full knowledge graph", "show full graph",
            )
        )
        matching_triples = query_graph(query, triples)
        if target_source:
            matching_triples = [t for t in matching_triples if t["source"] == target_source]
        selected_triples = [t for t in triples if not target_source or t.get("source") == target_source]
        debug_triples = matching_triples[:100] if matching_triples else selected_triples[:100]

        if is_overview_query:
            answer = format_graph_answer(
                triples,
                source_filter=target_source,
                limit=100,
            )
        else:
            if is_excel_relationship_query(query):
                excel_chunks = [c for c in get_excel_chunks() if not target_source or c["source"] == target_source]
                debug_chunks = excel_chunks
                relation_context = build_excel_relationship_context(excel_chunks)
                relation_context += "\n\n" + build_grounded_context(query, excel_chunks, source_filter=target_source)
                sources_used = [c["source"] for c in excel_chunks]
            else:
                results = search(query, top_k_per_source=6, source_filter=target_source)
                debug_chunks = results
                relation_context = build_grounded_context(query, results, source_filter=target_source)
                sources_used = [r["source"] for r in results]

            if matching_triples:
                relation_context += "\n\nKNOWLEDGE GRAPH RELATIONSHIPS:\n" + "\n".join(
                    f"{t['subject']} --[{t['relation']}]--> {t['object']} (source: {t['source']})"
                    for t in matching_triples[:30]
                )

            if is_excel_relationship_query(query):
                relationship_report = relation_context.split("\n\nEXCEL WORKBOOK EVIDENCE:", 1)[0]
                answer = relationship_report + "\n\nDetailed interpretation:\n" + generate_answer(
                    relation_context, query
                )
            else:
                answer = generate_answer(relation_context, query)

            if not matching_triples:
                answer += "\n\n" + format_graph_answer(triples, source_filter=target_source)

        sources_used = sorted(set(sources_used) | {t.get("source", "") for t in (selected_triples if is_overview_query else matching_triples)})

    elif intent == "SUMMARY":
        if is_excel_query(query):
            excel_chunks = get_excel_chunks()
            workbook_sources = sorted(set(c["source"] for c in excel_chunks))
            target_source = (
                None if asks_for_all_excel_files(query)
                else find_mentioned_source(query, workbook_sources)
            )
            results = [c for c in excel_chunks if not target_source or c["source"] == target_source]
        elif is_image_query(query):
            image_chunks = get_image_chunks()
            image_sources = sorted(set(c["source"] for c in image_chunks))
            target_source = find_mentioned_source(query, image_sources)

            if target_source:
                results = [
                    chunk for chunk in image_chunks
                    if chunk["source"] == target_source
                ]
            elif len(image_sources) == 1:
                results = image_chunks
            else:
                visual_results = search(query, top_k_per_source=8)
                results = [
                    chunk for chunk in visual_results
                    if chunk.get("modality", chunk.get("type", "")).upper() == "IMAGE"
                ]
                results = results or image_chunks
        else:
            sources = get_known_sources()
            target_source = find_mentioned_source(query, sources)
            summary_limit = 32 if target_source and target_source.lower().endswith((".docx", ".pptx", ".ppt")) else 8
            results = get_summary_chunks(
                per_source_limit=summary_limit,
                source_filter=target_source,
            )
        context = build_grounded_context(query, results, source_filter=target_source)
        debug_chunks = results
        answer = generate_summary(context)
        sources_used = [r["source"] for r in results]

    else:  # NORMAL
        target_source = find_mentioned_source(query, get_known_sources())
        results = search(query, top_k_per_source=4, source_filter=target_source)
        debug_chunks = results
        context = build_grounded_context(query, results, source_filter=target_source)
        answer = generate_answer(context, query)
        sources_used = [r["source"] for r in results]

    save_answer_record(query, answer, sources_used)
    debug = {
        "retrieval_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "chunks_used": len(debug_chunks),
        "retrieved_chunks": [
            {
                "score": chunk.get("retrieval_score", chunk.get("score")),
                "source_type": chunk.get("modality", chunk.get("type", "TEXT")),
                "filename": chunk.get("source", "unknown"),
                "page": chunk.get("page"),
                "sheet": chunk.get("sheet"),
                "rows": (
                    f"{chunk['row_start']}-{chunk.get('row_end', chunk['row_start'])}"
                    if chunk.get("row_start") is not None else None
                ),
                "text": chunk.get("text", ""),
            }
            for chunk in debug_chunks
        ],
        "graph_triples": [
            {
                "subject": triple.get("subject", ""),
                "relation": triple.get("relation", ""),
                "object": triple.get("object", ""),
                "source": triple.get("source", ""),
            }
            for triple in debug_triples
        ],
    }
    return answer, sorted(set(sources_used)), debug


