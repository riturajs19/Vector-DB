import requests

OLLAMA_URL = "http://localhost:11434"

def is_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False

def embed(text: str, model: str = "nomic-embed-text") -> list[float]:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                      json={"model": model, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]

def generate(prompt: str, model: str = "llama3.2") -> str:
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=180)
    r.raise_for_status()
    return r.json()["response"]