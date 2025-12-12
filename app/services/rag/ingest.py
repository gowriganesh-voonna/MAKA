# app/services/rag/ingest.py

# The RAGIngestor class is the core document processing engine of your Maka RAG pipeline.
#  It transforms raw documents/text into searchable vector embeddings stored in ChromaDB for semantic retrieval.
import os
from typing import List, Dict
from app.services.rag.chunker import load_document, chunk_text
from sentence_transformers import SentenceTransformer
import uuid
import chromadb
from chromadb.config import Settings  # (not used for PersistentClient, safe import)
import chromadb

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class RAGIngestor:
    def __init__(self, persist_dir: str = "data/embeddings_rag"):
        os.makedirs(persist_dir, exist_ok=True)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        # use PersistentClient (Chroma 0.5+)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="rag_documents")

    def ingest_file(self, path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        text = load_document(path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        inserted = []
        for idx, chunk in enumerate(chunks):
            doc_id = f"{uuid.uuid4()}"
            embedding = self.embedder.encode(chunk).tolist()
            metadata = {"source": os.path.basename(path), "chunk_index": idx}
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata]
            )
            inserted.append({"id": doc_id, "metadata": metadata, "text": chunk})
        return inserted

    def ingest_text(self, text: str, source: str = "text-input", chunk_size: int = 500, overlap: int = 50):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        inserted = []
        for idx, chunk in enumerate(chunks):
            doc_id = f"{uuid.uuid4()}"
            embedding = self.embedder.encode(chunk).tolist()
            metadata = {"source": source, "chunk_index": idx}
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata]
            )
            inserted.append({"id": doc_id, "metadata": metadata, "text": chunk})
        return inserted
