import requests
import time
import subprocess
import sys

API = "http://localhost:8080"

def test_endpoints():
    print("1. Testing /status...")
    r = requests.get(f"{API}/status")
    assert r.status_code == 200, f"Status failed: {r.text}"
    print("   /status response:", r.json())

    print("2. Testing /items...")
    r = requests.get(f"{API}/items")
    assert r.status_code == 200
    items = r.json()
    print(f"   Preloaded demo items: {len(items)}")
    assert len(items) > 0, "No preloaded items found"

    print("3. Testing /search with HNSW, KD-Tree, and BruteForce...")
    for algo in ["hnsw", "kdtree", "bruteforce"]:
        r = requests.get(f"{API}/search", params={"v": "0.86,0.06,0.09,0.88", "k": 3, "algo": algo})
        assert r.status_code == 200, f"{algo} search failed: {r.text}"
        res = r.json().get("results", [])
        print(f"   Algo {algo} returned {len(res)} results. Top hit id={res[0]['id']} dist={res[0]['distance']:.4f}")

    print("4. Testing /insert (demo vector)...")
    r = requests.post(f"{API}/insert", json={
        "metadata": "Test Vector Item",
        "category": "other",
        "embedding": [0.5, 0.5, 0.5, 0.5]
    })
    assert r.status_code == 200, f"Insert failed: {r.text}"
    new_id = r.json()["id"]
    print(f"   Inserted demo vector id={new_id}")

    print("5. Testing /delete...")
    r = requests.delete(f"{API}/delete/{new_id}")
    assert r.status_code == 200 and r.json().get("ok") is True
    print(f"   Deleted demo vector id={new_id}")

    print("6. Testing /doc/insert...")
    fake_768d = [0.01 * (i % 10) for i in range(768)]
    r = requests.post(f"{API}/doc/insert", json={
        "title": "Test Document",
        "text": "This is a test document text content.",
        "embedding": fake_768d
    })
    assert r.status_code == 200, f"Doc insert failed: {r.text}"
    doc_id = r.json()["id"]
    print(f"   Inserted document chunk id={doc_id}")

    print("7. Testing /doc/search...")
    r = requests.post(f"{API}/doc/search", json={
        "embedding": fake_768d,
        "k": 2
    })
    assert r.status_code == 200, f"Doc search failed: {r.text}"
    hits = r.json().get("results", [])
    assert len(hits) > 0, "No doc hits returned"
    print(f"   Doc search returned {len(hits)} hits. Top hit id={hits[0]['id']} dist={hits[0]['distance']:.4f}")

    print("8. Testing /doc/delete...")
    r = requests.delete(f"{API}/doc/delete/{doc_id}")
    assert r.status_code == 200 and r.json().get("ok") is True
    print(f"   Deleted document chunk id={doc_id}")

    print("\nAll VectorDB server endpoints passed successfully!")

if __name__ == "__main__":
    test_endpoints()
