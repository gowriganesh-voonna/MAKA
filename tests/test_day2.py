from app.services.memory.memory_service import MemoryService
from app.services.llm.gemini_service import GeminiLLM

memory = MemoryService()
llm = GeminiLLM()

# STEP 1: Add messages
memory.add_short_term("user", "Hi")
reply = llm.chat("Hello!")
memory.add_short_term("assistant", reply)

# STEP 2: Test memory recall
memory_list = memory.get_short_memory()
print("Short-Term Memory:", memory_list)

# STEP 3: Ask the model
prompt = "What was the user's last message?"
full_context = str(memory_list)
response = llm.chat(full_context + "\n\n" + prompt)

print("\nModel Response:", response)
