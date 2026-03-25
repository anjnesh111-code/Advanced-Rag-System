import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import re

_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower())


class HybridRetriever:

    def __init__(self):
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        self.faiss_index = None
        self.bm25_index = None
        self.embeddings = None

    def build_index(self, chunks, metadata=None):
        self.chunks = [c.strip() for c in chunks if isinstance(c, str) and c.strip()]

        if not self.chunks:
            raise ValueError("No valid chunks found for indexing")

        self.metadata = metadata or [{} for _ in self.chunks]

        embeddings = _embedder.encode(self.chunks, convert_to_numpy=True)

        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        self.embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)

        from rank_bm25 import BM25Okapi
        tokenized = [clean_text(doc).split() for doc in self.chunks]
        self.bm25_index = BM25Okapi(tokenized)

    def semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        query_embedding = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        tokenized_query = clean_text(query).split()
        scores = self.bm25_index.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def hybrid_search(self, query: str, top_k: int = 8, alpha: float = 0.6) -> List[Tuple[int, float, str]]:
        semantic_results = self.semantic_search(query, top_k * 3)
        bm25_results = self.bm25_search(query, top_k * 3)

        semantic_dict = {idx: score for idx, score in semantic_results}
        bm25_dict = {idx: score for idx, score in bm25_results}

        semantic_min = min(semantic_dict.values(), default=0)
        semantic_max = max(semantic_dict.values(), default=1)
        bm25_min = min(bm25_dict.values(), default=0)
        bm25_max = max(bm25_dict.values(), default=1)

        def normalize(val, mn, mx):
            if mx == mn:
                return 0.5
            return (val - mn) / (mx - mn)

        all_indices = set(semantic_dict.keys()) | set(bm25_dict.keys())
        combined = {}

        for idx in all_indices:
            s_score = normalize(semantic_dict.get(idx, semantic_min), semantic_min, semantic_max)
            b_score = normalize(bm25_dict.get(idx, bm25_min), bm25_min, bm25_max)
            combined[idx] = alpha * s_score + (1 - alpha) * b_score

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        final = []
        seen = set()

        for idx, score in sorted_results:
            text = self.chunks[idx]

            if text not in seen:
                seen.add(text)
                final.append((idx, score, text))

            if len(final) >= top_k:
                break

        return final

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        results = self.hybrid_search(query, top_k=top_k)
        return [r[2] for r in results]

    def boost_documents(self, doc_indices: List[int], boost_factor: float = 1.2):
        for idx in doc_indices:
            if 0 <= idx < len(self.embeddings):
                self.embeddings[idx] *= boost_factor

        boosted = self.embeddings.copy().astype(np.float32)
        faiss.normalize_L2(boosted)

        dim = boosted.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(boosted)