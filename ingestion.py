import os
import re
import json
from typing import List, Dict, Tuple
from pathlib import Path


def load_text_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_json_file(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        return [str(v) for v in data.values()]
    return [str(data)]


def load_documents_from_directory(directory: str) -> List[Dict]:
    documents = []
    supported_extensions = {".txt", ".md", ".json"}
    
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() not in supported_extensions:
            continue
        try:
            if path.suffix.lower() == ".json":
                texts = load_json_file(str(path))
                for i, text in enumerate(texts):
                    documents.append({
                        "text": text,
                        "source": str(path),
                        "doc_id": f"{path.stem}_{i}"
                    })
            else:
                text = load_text_file(str(path))
                documents.append({
                    "text": text,
                    "source": str(path),
                    "doc_id": path.stem
                })
        except Exception as e:
            print(f"  Warning: Could not load {path}: {e}")
    
    return documents


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'-{3,}', '---', text)
    return text.strip()


def chunk_by_sentence(text: str, max_chunk_size: int = 300, overlap: int = 50) -> List[str]:
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_words = []
    
    for sentence in sentences:
        sentence_words = sentence.split()
        
        if len(current_words) + len(sentence_words) > max_chunk_size and current_words:
            chunks.append(" ".join(current_words))
            overlap_words = current_words[-overlap:] if len(current_words) > overlap else current_words
            current_words = overlap_words + sentence_words
        else:
            current_words.extend(sentence_words)
    
    if current_words:
        chunks.append(" ".join(current_words))
    
    return [c for c in chunks if len(c.split()) >= 10]


def chunk_by_fixed_size(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 10:
            chunks.append(chunk)
    
    return chunks


def chunk_by_paragraph(text: str, max_chunk_size: int = 400) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para.split())
        
        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return [c for c in chunks if len(c.split()) >= 10]


def ingest_documents(
    documents: List[Dict],
    chunk_strategy: str = "sentence",
    chunk_size: int = 250,
    overlap: int = 40
) -> Tuple[List[str], List[Dict]]:
    all_chunks = []
    all_metadata = []
    
    strategy_map = {
        "sentence": lambda t: chunk_by_sentence(t, chunk_size, overlap),
        "fixed": lambda t: chunk_by_fixed_size(t, chunk_size, overlap),
        "paragraph": lambda t: chunk_by_paragraph(t, chunk_size),
    }
    
    chunk_fn = strategy_map.get(chunk_strategy, strategy_map["sentence"])
    
    for doc in documents:
        cleaned = clean_text(doc["text"])
        chunks = chunk_fn(cleaned)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": doc.get("source", "unknown"),
                "doc_id": doc.get("doc_id", "unknown"),
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
    
    return all_chunks, all_metadata


def ingest_raw_texts(
    texts: List[str],
    chunk_strategy: str = "sentence",
    chunk_size: int = 250,
    overlap: int = 40
) -> Tuple[List[str], List[Dict]]:
    documents = [{"text": t, "source": "inline", "doc_id": f"doc_{i}"} for i, t in enumerate(texts)]
    return ingest_documents(documents, chunk_strategy, chunk_size, overlap)


def save_chunks(chunks: List[str], metadata: List[Dict], output_path: str = "data/chunks.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = [{"chunk": c, "metadata": m} for c, m in zip(chunks, metadata)]
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: str = "data/chunks.json") -> Tuple[List[str], List[Dict]]:
    with open(input_path, "r") as f:
        payload = json.load(f)
    chunks = [item["chunk"] for item in payload]
    metadata = [item["metadata"] for item in payload]
    return chunks, metadata
