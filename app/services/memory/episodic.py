import json
from datetime import datetime
import os

class EpisodicMemory:
    """
    Stores events (episodes) with timestamps.
    """

    def __init__(self, file_path="data/memory/episodic_memory.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # create file if not exists
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump([], f)

    def store_event(self, event: str):
        timestamp = datetime.now().isoformat()
        record = {"timestamp": timestamp, "event": event}

        with open(self.file_path, "r+") as f:
            data = json.load(f)
            data.append(record)
            f.seek(0)
            json.dump(data, f, indent=4)

    def get_events(self):
        with open(self.file_path, "r") as f:
            return json.load(f)
