# 🚀 Self-Improving RAG System (Hackathon Project)

A production-ready Retrieval-Augmented Generation (RAG) system that improves context relevance, reduces hallucinations, and learns from past interactions.

Built from scratch using hybrid retrieval, reranking, memory, and Groq LLaMA 3.3.

---

## 🧠 Problem Statement

Traditional RAG systems:

* retrieve irrelevant context
* hallucinate answers
* fail on vague/complex queries

This system solves these using:

* advanced retrieval
* reranking
* confidence-based failure handling
* memory-based learning

---

## 🏗️ Architecture

```
User Query
   ↓
Query Rewriting + Decomposition
   ↓
Memory Context Injection
   ↓
Hybrid Retrieval (FAISS + BM25)
   ↓
CrossEncoder Reranking
   ↓
Context Optimization (diverse + compressed)
   ↓
Confidence Scoring
   ↓
   ├── Low Confidence → Fallback (no hallucination)
   ↓
LLM Generation (Groq - LLaMA 3.3 70B)
   ↓
Memory Storage (Self-learning)
   ↓
Final Answer + Confidence Score
```

---

## ⚙️ Features

### 🔍 Advanced Retrieval

* FAISS (semantic search)
* BM25 (keyword search)
* Hybrid fusion for better relevance

### 🧠 Query Understanding

* Query rewriting
* Complex query decomposition
* Intent-aware processing

### 📊 Ranking & Filtering

* CrossEncoder reranking (`ms-marco-MiniLM`)
* Relevance threshold filtering

### ⚡ Context Optimization

* Remove duplicate chunks
* Select diverse information
* Reduce noise before generation

### 💾 Memory System

* Stores past queries and answers
* Boosts frequently used documents
* Improves retrieval over time

### 🛡️ Hallucination Reduction

* Confidence scoring
* Fallback responses for low confidence
* Strict grounding in retrieved context

---

## 📊 Baseline vs Advanced System

| Feature               | Baseline RAG | This System         |
| --------------------- | ------------ | ------------------- |
| Retrieval             | FAISS only   | FAISS + BM25        |
| Query Handling        | Raw          | Rewrite + Decompose |
| Ranking               | None         | CrossEncoder        |
| Context               | Raw chunks   | Optimized           |
| Memory                | ❌            | ✅                   |
| Confidence            | ❌            | ✅                   |
| Hallucination Control | ❌            | ✅                   |

---

## 🧠 How It Works

1. Query is rewritten for clarity
2. Hybrid retrieval fetches relevant chunks
3. Reranker improves relevance
4. Context is optimized
5. Memory enhances query understanding
6. Groq LLaMA generates grounded response
7. Confidence score decides reliability

---

## 🛡️ Hallucination Reduction Strategy

* Context-grounded generation
* Reranking removes irrelevant data
* Confidence threshold prevents weak answers
* Fallback instead of guessing

---

## 💡 Self-Improvement

After each query:

* Interaction is stored
* Useful documents are boosted
* Similar queries use past knowledge

👉 System gets better over time

---

## ⚠️ Limitations

* Requires Groq API key
* Depends on knowledge base quality
* Initial latency due to model loading
* Memory is basic (can be improved)

---

## 🚀 Future Improvements

* Web search integration
* Long-term vector memory
* Streaming responses
* UI improvements
* Multi-modal support

---

## 🧪 Example

**Input:** What is RAG?

**Output:**

* Definition
* Explanation
* Example
* Applications

---

## 🧑‍💻 Tech Stack

* Python
* FAISS
* BM25
* Sentence Transformers
* CrossEncoder
* Groq API (LLaMA 3.3 70B)
* Gradio

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open:

```
http://127.0.0.1:7860
```

---

## 🔑 Environment Setup

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

## 🌟 Key Insight

This system doesn’t just generate answers.

👉 It retrieves → ranks → validates → learns → improves.
