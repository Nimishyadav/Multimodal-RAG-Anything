from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():
    """Loads a cross-encoder reranker once and reuses it.
    Tries the preferred model first, falls back to a smaller one
    if that one isn't available."""
    global _reranker
    if _reranker is None:
        try:
            _reranker = CrossEncoder("BAAI/bge-reranker-base")
        except Exception:
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query, chunks, top_k=6):
    """Re-scores retrieved chunks against the query using a cross-encoder -
    which looks at query+chunk TOGETHER (more accurate than comparing
    precomputed vectors) - and returns the best top_k, reordered."""
    if not chunks:
        return chunks

    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [c for c, s in scored[:top_k]]
