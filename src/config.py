# config.py
"""
Configuration file for Naive RAG Assignment.
All tunable parameters are stored here instead of hardcoding in code.
"""

# ---------------- Data ----------------
PASSAGES_PATH = "data/processed/passages_clean.parquet"

# Path to test queries dataset
QUERIES_PATH = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet"

# ---------------- Embeddings ----------------
# Default: MiniLM (384-dim, lightweight, good baseline)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
NORMALIZE = True

# ---------------- Vector Index ----------------
# Using FAISS 
TOP_K = 1   # naive_rag retrieval depth

# ---------------- LLM ----------------
# Gemini free-tier (cloud)
LLM = "gemini-2.5-flash"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
