import os
import gradio as gr
from pipeline import RAGPipeline
from confidence import get_confidence_label
from ingestion import load_documents_from_directory

rag = RAGPipeline()

documents = load_documents_from_directory("data")

if documents:
    rag.index_documents(documents)
else:
    rag.index_texts([
        "Machine learning enables systems to learn from data.",
        "Deep learning uses neural networks.",
        "Transformers use attention mechanisms.",
        "RAG improves answers using retrieval."
    ])


def ask(query):
    if not query.strip():
        return "Please enter a question."

    result = rag.run(query)
    label = get_confidence_label(result["confidence"])

    return f"{result['answer']}\n\nConfidence: {result['confidence']} ({label})"


demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(placeholder="Ask something..."),
    outputs=gr.Textbox(),
    title="🚀 Self-Improving RAG System",
    description="Hybrid RAG + Groq LLaMA 3.3 for accurate and intelligent responses"
)

demo.launch()