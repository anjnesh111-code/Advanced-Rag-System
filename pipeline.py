import time
from typing import List, Dict

from query import rewrite_query, decompose_complex_query
from retrieval import HybridRetriever
from reranker import rerank_chunks, filter_by_relevance_threshold
from optimizer import select_diverse_chunks
from generator import generate_answer
from memory import (
    store_interaction, retrieve_cached_answer,
    get_memory_context_for_query, store_high_quality_chunk
)
from confidence import (
    compute_retrieval_confidence, compute_answer_confidence,
    compute_combined_confidence, is_confidence_sufficient
)


class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.is_indexed = False

    def index_texts(self, texts: List[str]):
        from ingestion import ingest_raw_texts
        chunks, metadata = ingest_raw_texts(texts, chunk_size=400, overlap=80)
        self.retriever.build_index(chunks, metadata)
        self.is_indexed = True

    def index_documents(self, documents):
        from ingestion import ingest_documents
        chunks, metadata = ingest_documents(documents, chunk_size=400, overlap=80)
        self.retriever.build_index(chunks, metadata)
        self.is_indexed = True

    def run(self, user_query: str) -> Dict:
        start_time = time.time()

        cached = retrieve_cached_answer(user_query)
        if cached:
            return cached

        memory_context = get_memory_context_for_query(user_query)
        rewritten_query = rewrite_query(user_query, memory_context)

        sub_queries = decompose_complex_query(rewritten_query)

        all_results = []
        seen = set()

        for sq in sub_queries:
            for r in self.retriever.hybrid_search(sq, top_k=10):
                if r[0] not in seen:
                    seen.add(r[0])
                    all_results.append(r)

        ranked = rerank_chunks(rewritten_query, all_results, top_k=8)
        filtered = filter_by_relevance_threshold(ranked) or ranked[:3]

        optimized = select_diverse_chunks(filtered, rewritten_query, max_chunks=3)

        chunk_texts = [c[2] for c in optimized]

        hybrid_scores = [r[1] for r in all_results[:5]]
        rerank_scores = [r[1] for r in ranked[:3]]

        retrieval_conf = compute_retrieval_confidence(hybrid_scores, rerank_scores)

        if not is_confidence_sufficient(retrieval_conf):
                answer = generate_answer(
                        f"Answer this using your general knowledge: {user_query}",
                        []
                    )
                combined_conf = 0.5
        else:
            answer, _, _ = generate_answer(rewritten_query, chunk_texts)
            answer_conf = compute_answer_confidence(rewritten_query, answer, chunk_texts)
            combined_conf = compute_combined_confidence(retrieval_conf, answer_conf)

        store_interaction(user_query, rewritten_query,
                          [c[0] for c in optimized], answer, combined_conf)

        for c in optimized:
            store_high_quality_chunk(c[2], threshold=0.7, confidence=combined_conf)

        return {
            "answer": answer,
            "confidence": round(combined_conf, 4),
            "latency_ms": round((time.time() - start_time) * 1000, 1)
        }