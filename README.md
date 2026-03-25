---
title: Advanced RAG System
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.36.1"
python_version: "3.10"
app_file: app.py
pinned: false
---
# Advanced Self-Improving RAG System

A production-ready, modular Retrieval-Augmented Generation system with measurable improvements over a basic RAG baseline. Built with FAISS, BM25, cross-encoders, and Gemini.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Query Understanding    │  ← Rewrite, decompose, classify intent
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Memory Lookup          │  ← Check cache, retrieve past context
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Hybrid Retrieval       │  ← FAISS semantic + BM25 keyword fusion
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Cross-Encoder Rerank   │  ← Re-score with ms-marco cross-encoder
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Context Optimization   │  ← Deduplicate, compress, diversify
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Confidence Scoring     │  ← Combined retrieval + answer confidence
└─────────────────────────┘
    │
    ├── LOW confidence ──► Fallback response (no hallucination)
    │
    ▼
┌─────────────────────────┐
│   Answer Generation      │  ← Gemini with intent-aware prompting
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Memory Storage         │  ← Store interaction, boost relevant docs
└─────────────────────────┘
    │
    ▼
Final Answer + Confidence Label
```

---

## Improvements Over Baseline

| Feature | Baseline | Advanced |
|---|---|---|
| Retrieval | FAISS only | FAISS + BM25 hybrid fusion |
| Query processing | Raw query | Rewriting + decomposition + intent |
| Ranking | None | Cross-encoder reranking |
| Context | Top-k chunks as-is | Deduplicated + LLM-compressed |
| Confidence | Raw cosine score | Multi-signal composite score |
| Failure handling | Hallucination risk | Fallback on low confidence |
| Memory | None | Query cache + document boosting |
| Self-improvement | None | Past interactions improve future retrieval |

---

## Project Structure

```
rag_system/
├── data/                         # Stores memory.json, evaluation results
├── src/
│   ├── query.py                  # Query rewriting, decomposition, intent classification
│   ├── retrieval.py              # HybridRetriever: FAISS + BM25
│   ├── reranker.py               # Cross-encoder reranking
│   ├── optimizer.py              # Deduplication + context compression
│   ├── generator.py              # Gemini answer generation
│   ├── memory.py                 # Persistent memory + self-improvement
│   ├── confidence.py             # Multi-signal confidence scoring
│   ├── baseline/
│   │   └── baseline_rag.py       # Simple FAISS + Gemini baseline
│   └── evaluation/
│       └── evaluate.py           # Comparative evaluation pipeline
├── app.py                        # Main entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd rag_system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

---

## Running the System

### Demo mode (3 queries, side-by-side baseline vs advanced)
```bash
python app.py --mode demo
```

### Evaluation mode (full benchmark, saves results to data/)
```bash
python app.py --mode evaluate
```

### Single query
```bash
python app.py --mode query --query "How does backpropagation work?"
```

---

## Module Details

### `src/query.py`
- `rewrite_query()` — Uses Gemini to clarify and expand vague queries
- `decompose_complex_query()` — Splits multi-part queries into focused sub-queries
- `classify_query_intent()` — Routes to factual / analytical / procedural prompting style

### `src/retrieval.py`
- `HybridRetriever.build_index()` — Builds FAISS index + BM25 index simultaneously
- `HybridRetriever.hybrid_search()` — Min-max normalized score fusion with configurable alpha
- `HybridRetriever.boost_documents()` — Modifies embeddings based on memory feedback

### `src/reranker.py`
- `rerank_chunks()` — Cross-encoder scoring using `ms-marco-MiniLM-L-6-v2`
- `filter_by_relevance_threshold()` — Removes irrelevant chunks before generation

### `src/optimizer.py`
- `deduplicate_chunks()` — Cosine similarity deduplication at configurable threshold
- `compress_context()` — LLM summarization when context exceeds token budget

### `src/generator.py`
- `generate_answer()` — Intent-aware prompting, strictly grounded in context
- `generate_answer_with_citations()` — Numbered citation tracking

### `src/memory.py`
- `store_interaction()` — Persists queries, answers, chunk indices to `data/memory.json`
- `retrieve_cached_answer()` — MD5 hash-based exact query cache
- `get_high_value_doc_indices()` — Returns most-retrieved docs for boosting
- `get_memory_context_for_query()` — Finds semantically similar past Q&A pairs

### `src/confidence.py`
- `compute_retrieval_confidence()` — Combines hybrid scores + rerank scores
- `compute_answer_confidence()` — Measures context-answer lexical coverage
- `compute_combined_confidence()` — Weighted composite (0.6 retrieval + 0.4 answer)
- Threshold: 0.35 — below this, system returns fallback instead of risking hallucination

### `src/baseline/baseline_rag.py`
- FAISS semantic search only
- Direct Gemini generation with no query processing or reranking

### `src/evaluation/evaluate.py`
- Evaluates on 8 test queries across 3 metrics
- Uses Gemini as judge for answer quality scoring
- Outputs per-query deltas + aggregate improvement percentages
- Saves full results to `data/evaluation_results.json`

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Relevance** | Cosine similarity between query embedding and answer embedding |
| **Quality** | Gemini-as-judge score (0–1): accuracy, completeness, groundedness |
| **Confidence** | System's internal confidence in the retrieval + answer pipeline |

---

## Self-Improvement Mechanism

After each query, the system:
1. Stores the full interaction in `data/memory.json`
2. Increments retrieval counts for each document chunk used
3. On subsequent queries, high-count chunks have their FAISS embeddings boosted
4. Similar past Q&A pairs are prepended as memory context for query rewriting
5. High-confidence answers (≥ 0.7) are cached for instant future retrieval

This means the system measurably improves as it handles more queries — frequent topics get better retrieval, and repeated questions get instant cached answers.

---

## Requirements

- Python 3.10+
- Gemini API key (free tier works)
- ~2GB RAM for cross-encoder model loading
