import os, time, json, numpy as np, yaml
from google import genai
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def setup_llm(model_name="google/flan-t5-small"):
   #Load a lightweight local model for QA (default: flan-t5-small).Returns tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def now_ts():
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def l2_normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True)
    denom[denom == 0] = 1e-12
    return (v / denom).astype("float32")

def log_json(msg: dict, path="results/step2_log.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"ts": now_ts(), **msg}) + "\n")

