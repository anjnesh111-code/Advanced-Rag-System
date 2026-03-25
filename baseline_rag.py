import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from generator import generate_answer 

_embedder = SentenceTransformer("all-MiniLM-L6-v2")


class BaselineRAG:
    def __init__(self):
        self.chunks: List[str] = []
        self.faiss_index = None

    def build_index(self, chunks: List[str]):
        self.chunks = chunks
        embeddings = _embedder.encode(chunks, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        query_embedding = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, top_k)
        return [
            (int(idx), float(score), self.chunks[int(idx)])
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def generate(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        return generate_answer(query, context)

    def query(self, user_query: str) -> dict:
        retrieved = self.retrieve(user_query, top_k=5)
        chunk_texts = [r[2] for r in retrieved]
        retrieval_scores = [r[1] for r in retrieved]

        answer = self.generate(user_query, chunk_texts)

        avg_score = float(np.mean(retrieval_scores)) if retrieval_scores else 0.0

        return {
            "query": user_query,
            "answer": answer,
            "retrieved_chunks": chunk_texts,
            "retrieval_scores": retrieval_scores,
            "confidence": avg_score,
            "system": "baseline"
        }