from app.services.memory.short_term import ShortTermMemory
from app.services.memory.long_term import LongTermMemory
from app.services.memory.episodic import EpisodicMemory


class MemoryService:
    """
    Unified Memory Manager for Agentic AI.
    Combines: Short-term + Long-term + Episodic.
    """

    def __init__(self):
        self.short_term = ShortTermMemory(window_size=5)
        self.long_term = LongTermMemory()
        self.episodic = EpisodicMemory()

    # ---------- SHORT TERM ----------
    def add_short_term(self, role, content):
        self.short_term.add(role, content)

    def get_short_memory(self):
        return self.short_term.get_memory()

    # ---------- LONG TERM ----------
    def add_long_term(self, text, metadata=None):
        return self.long_term.add(text, metadata)

    def search_long_term(self, query, k=3):
        return self.long_term.search(query, k)

    # ---------- EPISODIC ----------
    def add_event(self, event: str):
        self.episodic.store_event(event)

    def get_events(self):
        return self.episodic.get_events()
