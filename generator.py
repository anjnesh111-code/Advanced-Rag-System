import os
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(query: str, chunks: list, intent: str = "factual"):
    client = _get_client()

    context = "\n\n".join(chunks[:5]) if chunks else ""

    if context:
        prompt = f"""
You are an expert AI assistant.

Answer in a clear, structured, and detailed way:

Include:
- Definition
- Explanation
- Example
- Applications

Question: {query}

Context:
{context}
"""
    else:
        prompt = f"""
You are an expert AI assistant.

Answer clearly and in detail:

Question: {query}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()

    except Exception as e:
        return f"GROQ ERROR: {str(e)}", 0.0, []

    confidence = 0.9 if chunks else 0.7

    return answer, confidence, list(range(min(5, len(chunks))))