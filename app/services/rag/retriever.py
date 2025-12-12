# app/services/rag/retriever.py
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class RAGRetriever:
    def __init__(self, persist_dir: str = "data/embeddings_rag"):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(name="rag_documents")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        q_emb = self.embedder.encode(query).tolist()
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        # results: dict with 'ids','embeddings','documents','metadatas'
        retrieved = []
        if not results or len(results.get("documents", [])) == 0:
            return retrieved

        # results are lists per query; we used single query index 0
        docs = results.get("documents", [])[0]
        metadatas = results.get("metadatas", [])[0]
        ids = results.get("ids", [])[0]

        for _id, doc, meta in zip(ids, docs, metadatas):
            retrieved.append({"id": _id, "text": doc, "metadata": meta})
        return retrieved
