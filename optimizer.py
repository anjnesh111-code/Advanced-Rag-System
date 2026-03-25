import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def deduplicate_chunks(
    chunks: List[Tuple[int, float, str]],
    similarity_threshold: float = 0.92
) -> List[Tuple[int, float, str]]:
    if not chunks:
        return []
    
    texts = [c[2] for c in chunks]
    embeddings = _embedder.encode(texts, convert_to_numpy=True)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    similarity_matrix = normalized @ normalized.T
    
    keep = []
    excluded = set()
    
    for i in range(len(chunks)):
        if i in excluded:
            continue
        keep.append(chunks[i])
        for j in range(i + 1, len(chunks)):
            if similarity_matrix[i][j] >= similarity_threshold:
                excluded.add(j)
    
    return keep


def compress_context(query: str, chunks: List[Tuple[int, float, str]], max_tokens: int = 800) -> str:
    combined_text = "\n\n".join([c[2] for c in chunks])
    
    if len(combined_text.split()) <= max_tokens:
        return combined_text
    
    return " ".join(combined_text.split()[:max_tokens])


def select_diverse_chunks(
    chunks: List[Tuple[int, float, str]],
    query: str,
    max_chunks: int = 5
) -> List[Tuple[int, float, str]]:
    deduplicated = deduplicate_chunks(chunks)
    return deduplicated[:max_chunks]