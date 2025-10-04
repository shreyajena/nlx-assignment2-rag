import os, faiss, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
import config
from utils import setup_llm
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

client, model_name = setup_llm()

# ---------- FAISS Utilities ----------
INDEX_PATH = "data/processed/faiss_index.bin"
SCHEMA_PATH = "data/processed/schema.parquet"
EMBED_PATH = "data/processed/embeddings.npy"

def build_faiss_index(passages_path, embed_model):
    # Build FAISS index from scratch and save schema + embeddings
    df = pd.read_parquet(passages_path)
    texts = df["passage"].tolist()

    embedder = SentenceTransformer(embed_model)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Save FAISS artifacts
    df["embedding"] = embeddings.tolist()
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(SCHEMA_PATH, index=False)
    np.save(EMBED_PATH, embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, INDEX_PATH)

    print(f"Built FAISS index with {len(df)} passages")
    return df, index, embedder

def load_faiss_index(embed_model):
    #Reload FAISS index + schema from disk to reload
    df = pd.read_parquet(SCHEMA_PATH)
    embeddings = np.load(EMBED_PATH)
    index = faiss.read_index(INDEX_PATH)
    embedder = SentenceTransformer(embed_model)
    print(f"Reloaded FAISS index with {len(df)} passages")
    return df, index, embedder

def get_or_build_faiss(passages_path, embed_model, force_rebuild=False):
    # If saved index exists, reload. Else rebuild from scratch
    if (not force_rebuild 
        and os.path.exists(INDEX_PATH) 
        and os.path.exists(SCHEMA_PATH)):
        return load_faiss_index(embed_model)
    else:
        return build_faiss_index(passages_path, embed_model)

# ---------- RAG Functions ----------
def retrieve(query, index, embedder, schema, k=config.TOP_K):
    # Retrieve top-k passages for a query
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb.astype("float32"), k)
    return [schema.iloc[i]["passage"] for i in I[0]]

def generate(query, context, style="basic"):
    # Generate an answer with LLM given query + context + different prompt
    # Saw that test data answers are usually short and so tried to prompt for balanced answer length
    if style == "cot":
        system_prompt = (
        "Reason step by step using only the provided context. "
        "At the end, give the final answer as a short phrase or number only."
    )
    elif style == "persona":  
        system_prompt = (
        "You are a helpful history teacher. Provide one short explanatory sentence, "
        "then give the final answer in 1–5 words only."
    )
    elif style == "instruction":  
        system_prompt = (
        "Answer the question clearly and directly using only the provided context. "
        "Respond in the fewest words possible (1–5 words)."
    )
    else:  
        system_prompt = (
        "Answer concisely using only the provided context. "
    )


    prompt = f"""{system_prompt}
    Context: {context}
    Question: {query}
    Answer:
    """
    try:
        tokenizer, llm = setup_llm() 
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = llm.generate(**inputs, max_new_tokens=50, do_sample=False, num_beams=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        return f"LLM Error: {e}"


def rag(query, schema, index, embedder, k=config.TOP_K, style="basic", concat=False):
    #Full RAG pipeline: retrieve + generate
    # Added context joining when retrieving top-k with k>1
   context = retrieve(query, index, embedder, schema, k)
   if not context:
        return "insufficient context"
   rag_context = "\n\n".join(context) if concat else context[0]
   return generate(query, rag_context, style=style)

# ---------- Main for Step 2 ----------
if __name__ == "__main__":
    schema, index, embedder = get_or_build_faiss(config.PASSAGES_PATH, config.EMBED_MODEL)
    q = "Where is Uruguay located?"
    print("Q:", q)
    print("A:", rag(q, schema, index, embedder))
