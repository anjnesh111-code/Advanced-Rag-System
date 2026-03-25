def rewrite_query(original_query: str, memory_context: str = "") -> str:
    base = f"Explain clearly with examples: {original_query} in AI, ML or Deep Learning context"
    if memory_context:
        return base + " using previous related knowledge"
    return base


def decompose_complex_query(query: str) -> list:
    if " and " in query:
        return [q.strip() for q in query.split(" and ") if q.strip()]
    return [query]


def classify_query_intent(query: str) -> str:
    q = query.lower()
    if "compare" in q or "difference" in q:
        return "comparison"
    if "how" in q or "why" in q:
        return "explanatory"
    return "factual"