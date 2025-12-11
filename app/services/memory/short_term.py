class ShortTermMemory:
    """
    Stores last N conversation messages.
    Format: { "role": "user/assistant", "content": "text" }
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.messages = []

    def add(self, role: str, content: str):
        """Add a message to memory."""
        self.messages.append({"role": role, "content": content})
        # keep last N messages only
        self.messages = self.messages[-self.window_size:]

    def get_memory(self):
        """Return all short-term messages."""
        return self.messages
