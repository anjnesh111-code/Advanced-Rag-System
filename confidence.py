from typing import List, Tuple
import numpy as np

CONFIDENCE_THRESHOLD = 0.35


def compute_retrieval_confidence(
    hybrid_scores: List[float],
    rerank_scores: List[float]
) -> float:
    if not hybrid_scores or not rerank_scores:
        return 0.0
    
    avg_hybrid = np.mean(hybrid_scores[:3]) if len(hybrid_scores) >= 3 else np.mean(hybrid_scores)
    
    top_rerank = rerank_scores[0] if rerank_scores else -10.0
    rerank_normalized = 1.0 / (1.0 + np.exp(-0.1 * top_rerank))
    
    num_results_score = min(len(hybrid_scores) / 5.0, 1.0)
    
    confidence = 0.4 * avg_hybrid + 0.4 * rerank_normalized + 0.2 * num_results_score
    return float(np.clip(confidence, 0.0, 1.0))


def compute_answer_confidence(
    query: str,
    answer: str,
    context_chunks: List[str]
) -> float:
    if not answer or not context_chunks:
        return 0.0
    
    answer_words = set(answer.lower().split())
    
    coverage_scores = []
    for chunk in context_chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
        coverage_scores.append(overlap)
    
    max_coverage = max(coverage_scores) if coverage_scores else 0.0
    
    uncertainty_phrases = [
        "i don't know", "i cannot", "not enough information",
        "unable to", "no information", "cannot determine"
    ]
    uncertainty_penalty = sum(1 for phrase in uncertainty_phrases if phrase in answer.lower()) * 0.15
    
    answer_length_score = min(len(answer.split()) / 50.0, 1.0)
    
    confidence = 0.5 * max_coverage + 0.3 * answer_length_score - uncertainty_penalty
    return float(np.clip(confidence, 0.0, 1.0))


def compute_combined_confidence(
    retrieval_confidence: float,
    answer_confidence: float
) -> float:
    return float(np.clip(0.6 * retrieval_confidence + 0.4 * answer_confidence, 0.0, 1.0))


def is_confidence_sufficient(confidence: float) -> bool:
    return confidence >= CONFIDENCE_THRESHOLD


def get_confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "HIGH"
    elif confidence >= 0.45:
        return "MEDIUM"
    elif confidence >= CONFIDENCE_THRESHOLD:
        return "LOW"
    else:
        return "INSUFFICIENT"
