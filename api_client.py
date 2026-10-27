import requests

API = "http://localhost:8080"


class SearchHit:
    def __init__(self, d):
        self.distance = float(d.get("distance", 0.0))
        self.id = int(d.get("id", 0))
        self.metadata = d.get("metadata", "")
        self.category = d.get("category", "")

    def __iter__(self):
        yield self.distance
        yield self.id

    def __getitem__(self, idx):
        return (self.distance, self.id, self.metadata, self.category)[idx]

    def __repr__(self):
        return f"SearchHit(id={self.id}, distance={self.distance:.4f}, metadata={self.metadata!r})"


class VectorDB:
    def insert(self, meta, cat, emb):
        r = requests.post(f"{API}/insert",
                          json={"metadata": meta, "category": cat, "embedding": emb})
        r.raise_for_status()
        return r.json()["id"]

    def remove(self, doc_id):
        r = requests.delete(f"{API}/delete/{doc_id}")
        r.raise_for_status()
        return r.json()["ok"]

    def search(self, q, k, algo):
        params = {"v": ",".join(str(x) for x in q), "k": k, "algo": algo}
        r = requests.get(f"{API}/search", params=params)
        r.raise_for_status()
        return [SearchHit(item) for item in r.json().get("results", [])]

    def all(self):
        r = requests.get(f"{API}/items")
        r.raise_for_status()
        return r.json()

    def size(self):
        try:
            r = requests.get(f"{API}/status", timeout=2)
            if r.status_code == 200:
                return r.json().get("demoCount", len(self.all()))
        except Exception:
            pass
        return len(self.all())

    def seed(self):
        r = requests.post(f"{API}/seed")
        r.raise_for_status()
        return r.json()


class DocItem:
    def __init__(self, d):
        self.id = d["id"]
        self.title = d["title"]
        self.text = d["text"]


class DocumentDB:
    def insert(self, title, text, emb):
        r = requests.post(f"{API}/doc/insert",
                          json={"title": title, "text": text, "embedding": emb})
        r.raise_for_status()
        return r.json()["id"]

    def search(self, emb, k):
        r = requests.post(f"{API}/doc/search", json={"embedding": emb, "k": k})
        r.raise_for_status()
        return [(item["distance"], DocItem(item)) for item in r.json().get("results", [])]

    def remove(self, doc_id):
        r = requests.delete(f"{API}/doc/delete/{doc_id}")
        r.raise_for_status()
        return r.json()["ok"]

    def all(self):
        r = requests.get(f"{API}/doc/list")
        r.raise_for_status()
        return [DocItem(d) for d in r.json()]

    def size(self):
        try:
            r = requests.get(f"{API}/status", timeout=2)
            if r.status_code == 200:
                return r.json().get("docCount", len(self.all()))
        except Exception:
            pass
        return len(self.all())
