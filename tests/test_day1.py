from app.services.llm.gemini_service import GeminiLLM

llm = GeminiLLM()
print(llm.chat("Hello! This is a Day-1 Gemini test."))
