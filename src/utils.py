import os, time, json, numpy as np, yaml
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

def setup_llm(api_key_env=api_key, model_name="gemini-2.5-flash"):
    if not api_key_env:
        raise RuntimeError(f"{api_key_env} not set. export {api_key_env}='YOUR_KEY'")
    client = genai.Client(api_key=api_key_env)
    return client, model_name

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

