import os, numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from naive_rag import retrieve, generate, get_or_build_faiss
import config
from utils import setup_llm
import os
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


tokenizer, llm = setup_llm() 

# ---------- Reranking ----------
def rerank_passages(query, passages, embedder, top_n=3):
    if not passages:
        return []
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    P = embedder.encode(passages, convert_to_numpy=True, show_progress_bar=False)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    scores = (P @ q[0])              # cosine via dot of normalized vectors
    idx = np.argsort(scores)[::-1][:top_n]
    return [passages[i] for i in idx]

# ---------- Enhanced Generate (for Step 5) ----------
def enhanced_generate(context, query, tokenizer, llm, style="basic"):
    """
    A cleaner generation function for grounded RAG.
    Ensures no duplicated labels and enforces citation-based answering.
    """
    prompt = (
        "Use ONLY the context below to answer the question.\n"
        "Cite supporting passages as [Doc:X]. "
        "If the answer cannot be found, reply 'Not found in context.'\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=2)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer



# ---------- Enhanced RAG ----------
def enhanced_rag(query, schema, index, embedder, k=3, style="basic_enhanced", return_contexts=False):
    """
    Enhanced RAG = Reranking + Grounded Citations
    """
    
    # Retrieve with document IDs (schema should have doc_id/text)
    retrieved = retrieve(query, index, embedder, schema, k)
    docs = [f"[Doc:{i+1}] {text}" for i, text in enumerate(retrieved)]

    # Rerank
    reranked = rerank_passages(query, retrieved, embedder, top_n=min(3, len(retrieved)))

    context = "\n\n".join([f"[Doc:{i+1}] {p}" for i, p in enumerate(reranked)])

    # Call generate() using new 'basic_enhanced' style - This has Grounded instrcutions Added as a style itself
    answer = generate(query, context, style="basic_enhanced")

    # --- optional context return for evaluation ---
    if return_contexts:
       return reranked, answer
    return answer

# ---------- Main - Running for quick testing ----------
if __name__ == "__main__":
    schema, index, embedder = get_or_build_faiss(config.PASSAGES_PATH, config.EMBED_MODEL)
    q ="How many long was Lincoln's formal education?"
    print("\nEnhanced RAG Answer:")
    print(enhanced_rag(q, schema, index, embedder))
