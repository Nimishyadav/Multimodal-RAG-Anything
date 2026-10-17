def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping chunks so context isn't lost at boundaries."""
    chunks = []
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size.")
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  

    return chunks


def chunk_segment(segment, source, segment_index, chunk_size=500, overlap=50):
    """Create provenance-preserving chunks for the shared content list."""
    chunks = chunk_text(segment["text"], chunk_size=chunk_size, overlap=overlap)
    if not chunks and segment.get("graph_triples"):
        chunks = ["\n".join(
            f"{triple['subject']} --[{triple['relation']}]--> {triple['object']}"
            for triple in segment["graph_triples"]
        )]
    modality = segment.get("modality", segment.get("type", "TEXT")).upper()
    page = segment.get("page")
    segment_id = segment.get("id", f"{source}:segment:{segment_index}")

    records = [
        {
            "id": f"{segment_id}:chunk:{chunk_index}",
            "parent_id": segment_id,
            "text": text,
            "source": source,
            "type": modality,
            "modality": modality,
            "page": page,
            "sheet": segment.get("sheet"),
            "row_start": segment.get("row_start"),
            "row_end": segment.get("row_end"),
        }
        for chunk_index, text in enumerate(chunks)
    ]

    if records and segment.get("graph_triples"):
        records[0]["graph_triples"] = segment["graph_triples"]

    return records
