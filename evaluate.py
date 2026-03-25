import os
import sys
import json
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_model = genai.GenerativeModel("gemini-2.5-flash")

TEST_QUERIES = [
    "What is machine learning?",
    "How does neural network training work?",
    "What are transformers in NLP?",
    "Explain the attention mechanism",
    "What is retrieval augmented generation?",
    "How does FAISS work for similarity search?",
    "What is the difference between supervised and unsupervised learning?",
    "How does backpropagation work?",
]


def compute_semantic_relevance(query: str, answer: str) -> float:
    if not answer or not query:
        return 0.0
    
    query_emb = _embedder.encode([query], convert_to_numpy=True)
    answer_emb = _embedder.encode([answer[:500]], convert_to_numpy=True)
    
    similarity = float(np.dot(query_emb[0], answer_emb[0]) / (
        np.linalg.norm(query_emb[0]) * np.linalg.norm(answer_emb[0]) + 1e-10
    ))
    return max(0.0, similarity)


def compute_answer_quality(query: str, answer: str, context_chunks: List[str]) -> float:
    prompt = f"""Rate the quality of this answer on a scale of 0.0 to 1.0.
Consider: accuracy, completeness, relevance to the question, and groundedness in context.
Return ONLY a decimal number between 0.0 and 1.0, nothing else.

Question: {query}
Context available: {' '.join(context_chunks[:2])[:300]}
Answer: {answer[:400]}

Quality score (0.0-1.0):"""
    
    try:
        response = _model.generate_content(prompt)
        score = float(response.text.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def evaluate_single_query(query: str, baseline_result: dict, advanced_result: dict) -> dict:
    baseline_relevance = compute_semantic_relevance(query, baseline_result.get("answer", ""))
    advanced_relevance = compute_semantic_relevance(query, advanced_result.get("answer", ""))
    
    baseline_quality = compute_answer_quality(
        query,
        baseline_result.get("answer", ""),
        baseline_result.get("retrieved_chunks", [])
    )
    advanced_quality = compute_answer_quality(
        query,
        advanced_result.get("answer", ""),
        [c[2] if isinstance(c, tuple) else c for c in advanced_result.get("retrieved_chunks", [])][:3]
    )
    
    return {
        "query": query,
        "baseline": {
            "relevance": round(baseline_relevance, 4),
            "quality": round(baseline_quality, 4),
            "confidence": round(baseline_result.get("confidence", 0.0), 4),
        },
        "advanced": {
            "relevance": round(advanced_relevance, 4),
            "quality": round(advanced_quality, 4),
            "confidence": round(advanced_result.get("confidence", 0.0), 4),
        },
        "improvements": {
            "relevance_delta": round(advanced_relevance - baseline_relevance, 4),
            "quality_delta": round(advanced_quality - baseline_quality, 4),
            "confidence_delta": round(
                advanced_result.get("confidence", 0.0) - baseline_result.get("confidence", 0.0), 4
            ),
        }
    }


def run_evaluation(baseline_rag, advanced_rag, queries: List[str] = None) -> dict:
    queries = queries or TEST_QUERIES
    results = []
    
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Query: {query[:60]}")
        
        print("  → Running baseline...")
        baseline_result = baseline_rag.query(query)
        
        print("  → Running advanced system...")
        advanced_result = advanced_rag.query(query)
        
        eval_result = evaluate_single_query(query, baseline_result, advanced_result)
        results.append(eval_result)
        
        print(f"  Baseline  → Relevance: {eval_result['baseline']['relevance']:.3f} | Quality: {eval_result['baseline']['quality']:.3f} | Confidence: {eval_result['baseline']['confidence']:.3f}")
        print(f"  Advanced  → Relevance: {eval_result['advanced']['relevance']:.3f} | Quality: {eval_result['advanced']['quality']:.3f} | Confidence: {eval_result['advanced']['confidence']:.3f}")
        print(f"  Delta     → Rel: {eval_result['improvements']['relevance_delta']:+.3f} | Qual: {eval_result['improvements']['quality_delta']:+.3f} | Conf: {eval_result['improvements']['confidence_delta']:+.3f}")
    
    summary = compute_summary(results)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Queries evaluated: {summary['total_queries']}")
    print(f"\nBaseline Averages:")
    print(f"  Relevance:  {summary['baseline_avg']['relevance']:.4f}")
    print(f"  Quality:    {summary['baseline_avg']['quality']:.4f}")
    print(f"  Confidence: {summary['baseline_avg']['confidence']:.4f}")
    print(f"\nAdvanced Averages:")
    print(f"  Relevance:  {summary['advanced_avg']['relevance']:.4f}")
    print(f"  Quality:    {summary['advanced_avg']['quality']:.4f}")
    print(f"  Confidence: {summary['advanced_avg']['confidence']:.4f}")
    print(f"\nImprovement:")
    print(f"  Relevance:  {summary['avg_improvements']['relevance']:+.4f} ({summary['avg_improvements']['relevance_pct']:+.1f}%)")
    print(f"  Quality:    {summary['avg_improvements']['quality']:+.4f} ({summary['avg_improvements']['quality_pct']:+.1f}%)")
    print(f"  Confidence: {summary['avg_improvements']['confidence']:+.4f} ({summary['avg_improvements']['confidence_pct']:+.1f}%)")
    print("="*60)
    
    output = {"results": results, "summary": summary}
    
    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nFull results saved to data/evaluation_results.json")
    return output


def compute_summary(results: List[dict]) -> dict:
    def avg(key1, key2):
        return np.mean([r[key1][key2] for r in results])
    
    baseline_relevance = avg("baseline", "relevance")
    baseline_quality = avg("baseline", "quality")
    baseline_confidence = avg("baseline", "confidence")
    advanced_relevance = avg("advanced", "relevance")
    advanced_quality = avg("advanced", "quality")
    advanced_confidence = avg("advanced", "confidence")
    
    def pct_change(old, new):
        return ((new - old) / max(old, 1e-10)) * 100
    
    return {
        "total_queries": len(results),
        "baseline_avg": {
            "relevance": round(float(baseline_relevance), 4),
            "quality": round(float(baseline_quality), 4),
            "confidence": round(float(baseline_confidence), 4),
        },
        "advanced_avg": {
            "relevance": round(float(advanced_relevance), 4),
            "quality": round(float(advanced_quality), 4),
            "confidence": round(float(advanced_confidence), 4),
        },
        "avg_improvements": {
            "relevance": round(float(advanced_relevance - baseline_relevance), 4),
            "relevance_pct": round(pct_change(baseline_relevance, advanced_relevance), 1),
            "quality": round(float(advanced_quality - baseline_quality), 4),
            "quality_pct": round(pct_change(baseline_quality, advanced_quality), 1),
            "confidence": round(float(advanced_confidence - baseline_confidence), 4),
            "confidence_pct": round(pct_change(baseline_confidence, advanced_confidence), 1),
        }
    }
