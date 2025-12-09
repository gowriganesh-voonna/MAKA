import google.generativeai as genai
from app.core.config import settings
from app.core.constants import GEMINI_MODEL

class GeminiLLM:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = GEMINI_MODEL

    def chat(self, prompt: str):
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        return response.text
