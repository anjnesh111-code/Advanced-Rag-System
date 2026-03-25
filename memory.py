import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

MEMORY_FILE = "data/memory.json"


def _load_memory() -> Dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {
        "query_history": [],
        "doc_boost_counts": {},
        "cached_answers": {},
        "query_patterns": {}
    }


def _save_memory(memory: Dict):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def _hash_query(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def store_interaction(
    original_query: str,
    rewritten_query: str,
    retrieved_chunk_indices: List[int],
    answer: str,
    confidence: float
):
    memory = _load_memory()
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "chunk_indices": retrieved_chunk_indices,
        "answer": answer,
        "confidence": confidence
    }
    memory["query_history"].append(entry)
    
    for idx in retrieved_chunk_indices:
        key = str(idx)
        memory["doc_boost_counts"][key] = memory["doc_boost_counts"].get(key, 0) + 1
    
    if confidence >= 0.7:
        query_hash = _hash_query(original_query)
        memory["cached_answers"][query_hash] = {
            "answer": answer,
            "rewritten_query": rewritten_query,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    memory["query_history"] = memory["query_history"][-500:]
    
    _save_memory(memory)

def store_high_quality_chunk(chunk: str, threshold: float, confidence: float):
    if confidence < threshold:
        return

    path = "data/learned.txt"

    with open(path, "a", encoding="utf-8") as f:
        f.write(chunk + "\n\n")
        
def retrieve_cached_answer(query: str) -> Optional[Dict]:
    memory = _load_memory()
    query_hash = _hash_query(query)
    return memory["cached_answers"].get(query_hash)


def get_high_value_doc_indices(top_n: int = 20) -> List[int]:
    memory = _load_memory()
    boost_counts = memory.get("doc_boost_counts", {})
    
    sorted_docs = sorted(boost_counts.items(), key=lambda x: x[1], reverse=True)
    return [int(idx) for idx, _ in sorted_docs[:top_n]]


def get_memory_context_for_query(query: str, top_k: int = 3) -> str:
    memory = _load_memory()
    history = memory.get("query_history", [])
    
    if not history:
        return ""
    
    query_words = set(query.lower().split())
    scored_entries = []
    
    for entry in history[-100:]:
        past_words = set(entry.get("original_query", "").lower().split())
        overlap = len(query_words & past_words) / max(len(query_words | past_words), 1)
        if overlap > 0.3:
            scored_entries.append((overlap, entry))
    
    scored_entries.sort(key=lambda x: x[0], reverse=True)
    top_entries = scored_entries[:top_k]
    
    if not top_entries:
        return ""
    
    context_parts = []
    for _, entry in top_entries:
        context_parts.append(f"Past Q: {entry['original_query']} | A: {entry['answer'][:200]}")
    
    return "\n".join(context_parts)


def get_memory_stats() -> Dict:
    memory = _load_memory()
    return {
        "total_interactions": len(memory.get("query_history", [])),
        "cached_answers": len(memory.get("cached_answers", {})),
        "boosted_documents": len(memory.get("doc_boost_counts", {}))
    }
