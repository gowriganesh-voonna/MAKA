# app/services/rag/rag_pipeline.py
from typing import List, Dict
from app.services.rag.retriever import RAGRetriever
from app.services.llm.gemini_service import GeminiLLM
from app.core.config import settings

DEFAULT_TOP_K = 3


def build_prompt_with_context(query: str, contexts: List[Dict]) -> str:
    """
    Build a prompt that includes retrieved contexts as numbered citations.
    """
    ctx_lines = []
    for i, ctx in enumerate(contexts, start=1):
        src = ctx.get("metadata", {}).get("source", "unknown")
        idx = ctx.get("metadata", {}).get("chunk_index", -1)
        ctx_lines.append(f"[{i}] (source: {src}, chunk: {idx})\n{ctx['text']}\n")
    context_block = "\n---\n".join(ctx_lines) if ctx_lines else ""
    prompt = (
        f"You are an assistant that answers questions using the provided context snippets. "
        f"Always cite the snippet number when you reference facts.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely and include the citation numbers in brackets when referencing context, "
        f"for example: (see [1]). If answer not found in context, say 'I don't know based on the provided documents.'"
    )
    return prompt


class RAGPipeline:
    def __init__(self, persist_dir: str = "data/embeddings_rag"):
        self.retriever = RAGRetriever(persist_dir=persist_dir)
        self.gemini_key = settings.GEMINI_API_KEY or ""
        # do not instantiate GeminiLLM if no key; handle gracefully
        self.llm = GeminiLLM() if self.gemini_key else None

    def answer(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        contexts = self.retriever.retrieve(query, top_k)
        prompt = build_prompt_with_context(query, contexts)
        response_text = None
        if self.llm:
            try:
                response_text = self.llm.chat(prompt)
            except Exception as e:
                response_text = f"[LLM error] {e}"
        else:
            # No API key — return contexts for inspection
            response_text = "No Gemini API key set. Retrieved contexts returned instead."

        return {"answer": response_text, "retrieved": contexts}
