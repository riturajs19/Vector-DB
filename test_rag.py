import requests
import time
import rag
import api_client as vc
import ollama_client as oc

def test_rag():
    print("Checking Ollama status...")
    if not oc.is_available():
        print("Ollama is not available, skipping RAG test.")
        return

    print("Ollama is online!")
    doc_db = vc.DocumentDB()

    sample_doc = (
        "Hierarchical Navigable Small World (HNSW) graphs are among the top-performing "
        "techniques for approximate nearest neighbor search. HNSW builds a multi-layer graph "
        "where the bottom layer contains all vectors and each upper layer contains an "
        "exponentially decreasing subset of vectors, providing logarithmic search complexity."
    )

    print("Inserting document via rag.insert_document...")
    ids = rag.insert_document(doc_db, "HNSW Graph Indexing", sample_doc)
    print(f"Document chunk IDs: {ids}")

    print("Testing rag.ask...")
    result = rag.ask(doc_db, "What is HNSW and what complexity does it achieve?", k=2)
    print("\n--- Answer from local LLM ---")
    print(result["answer"])
    print("\n--- Retrieved Contexts ---")
    for ctx in result["contexts"]:
        print(f"Title: {ctx['title']}, Distance: {ctx['distance']:.4f}")

    # Cleanup test document
    for doc_id in ids:
        doc_db.remove(doc_id)
    print("Cleaned up test document.")
    print("RAG Pipeline Verified Successfully!")

if __name__ == "__main__":
    test_rag()
