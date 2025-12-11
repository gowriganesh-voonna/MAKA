import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
import os

class LongTermMemory:
    """
    Embedding-based Vector Memory using new ChromaDB (0.5+) client.
    """

    def __init__(self, persist_dir="data/embeddings"):
        os.makedirs(persist_dir, exist_ok=True)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # NEW Chroma client (no deprecated Settings fields)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"}  # optional but valid
        )

    def add(self, text: str, metadata: dict = None):
        """Insert memory into vector DB."""
        embedding = self.embedder.encode(text).tolist()
        doc_id = str(uuid.uuid4())

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )

        return doc_id

    def search(self, query: str, k: int = 3):
        """Semantic search from long-term memory."""
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results
