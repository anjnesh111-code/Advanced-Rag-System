from sentence_transformers import CrossEncoder
from typing import List, Tuple

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    return _model


def rerank_chunks(query: str, chunks: List[Tuple[int, float, str]], top_k: int = 5):
    if not chunks:
        return []

    model = _get_model()

    pairs = [(query, c[2]) for c in chunks]
    scores = model.predict(pairs)

    scored = [(chunks[i][0], float(scores[i]), chunks[i][2]) for i in range(len(chunks))]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def filter_by_relevance_threshold(ranked_chunks, threshold: float = -2.0):
    return [c for c in ranked_chunks if c[1] >= threshold]