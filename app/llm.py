import os
from typing import Optional

try:
    from groq import Groq
except ImportError:
    Groq = None


SYSTEM_PROMPT = (
    "You answer short follow-up questions about an object the user is currently "
    "pointing a camera at. Use ONLY the provided context (Wikipedia summary and "
    "any text read off the object via OCR). Be concise (1-3 sentences). "
    "If the answer is not in the context, say so plainly."
)


class GroqClient:
    """Thin wrapper around Groq's chat API. Falls back gracefully when unavailable."""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.client: Optional[object] = None
        self.available = False

        if Groq is None:
            print("[llm] 'groq' package not installed - pip install groq to enable Q&A.")
            return
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[llm] GROQ_API_KEY not set - get a free key at https://console.groq.com")
            return
        try:
            self.client = Groq(api_key=api_key)
            self.available = True
            print(f"[llm] Groq ready (model: {self.model})")
        except Exception as e:
            print(f"[llm] could not initialize Groq client: {e}")

    def ask(self, question: str, details: dict) -> str:
        if not self.available or not question.strip():
            return "(LLM unavailable - set GROQ_API_KEY and install 'groq')"

        context_parts = []
        if details.get("title"):
            context_parts.append(f"Object: {details['title']}")
        if details.get("summary"):
            context_parts.append(f"Wikipedia summary: {details['summary']}")
        if details.get("ocr_text"):
            context_parts.append(f"Text visible on object: {details['ocr_text']}")
        context = "\n\n".join(context_parts) or "(no context available)"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(LLM error: {e})"
