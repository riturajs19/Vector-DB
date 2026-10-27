# In-Depth C++ Vector Database & Local RAG System

A high-performance Vector Database built from scratch in modern C++ (featuring **HNSW**, **KD-Tree**, and **Brute-Force** search algorithms) paired with a **Local RAG Pipeline** powered by Ollama and an interactive **Streamlit** user interface.

---

## 🏛️ Architecture Overview

```
                               ┌─────────────────────────────────┐
                               │   Streamlit Web UI (app.py)     │
                               │  (Demo Search | Docs | Ask AI)  │
                               └──────────────┬──────────────────┘
                                              │
                    ┌─────────────────────────┴────────────────────────┐
                    │                                                  │
                    ▼                                                  ▼
     ┌────────────────────────────┐                     ┌─────────────────────────────┐
     │     RAG Engine (rag.py)    │                     │  VectorDB Client            │
     └──────────────┬─────────────┘                     │  (api_client.py)            │
                    │                                   └──────────────┬──────────────┘
         ┌──────────┴──────────┐                                       │
         ▼                     ▼                                       │
┌────────────────┐   ┌───────────────────┐                             │
│  Ollama Client │   │ nomic-embed-text  │                             │
│ (llama3.2 LLM) │   │ (768D Embeddings) │                             │
└────────────────┘   └─────────┬─────────┘                             │
                               │                                       │
                               └───────────────────┬───────────────────┘
                                                   │ HTTP REST (port 8080)
                                                   ▼
                     ┌───────────────────────────────────────────────┐
                     │            C++ Vector Database Server         │
                     │                    (db.exe)                   │
                     ├───────────────────────────────────────────────┤
                     │  • HNSW Multilayer Graph (O(log N))           │
                     │  • Balanced KD-Tree (median split)            │
                     │  • Brute-Force Linear Scan Baseline           │
                     │  • Cosine, Euclidean & Manhattan metrics      │
                     │  • In-memory Thread-Safe Storage              │
                     └───────────────────────────────────────────────┘
```

---

## 🔍 Key Issues Identified & Fixed

1. **Missing `httplib.h`**:
   - `main.cpp` depended on `httplib.h`, but the header file was absent in the project folder, preventing compilation.
   - Restored and integrated `httplib.h`.

2. **Static Linking & Windows Socket Linking**:
   - Compiling without `-static` required MinGW UCRT64 runtime DLLs in the system `PATH`.
   - Linked statically (`-static -lws2_32`) so `db.exe` runs standalone on any Windows environment without external runtime dependencies.

3. **Dimension Mismatch in Demo Vectors**:
   - `main.cpp` hardcoded `DEMO_DIMS = 16`, but `app.py` sent 4-dimensional vectors by default (`0.9, 0.1, 0.87, 0.1`), causing `VectorDB::insert` to throw `400 Bad Request: dimension mismatch`.
   - Updated `VectorDB` to support dynamic dimension detection: dimensions are determined seamlessly on the first inserted vector, or dynamically adapt per store.
   - Pre-loaded representative 4D demo vectors across `hr`, `finance`, and `other` categories so the demo search is instantly functional upon startup.

4. **Fragile JSON Parsing in Server**:
   - Naive string search for keys like `"embedding"` failed whenever document text contained quotes, brackets, or the word `"embedding"`.
   - Implemented depth-aware bracket parsing and JSON escape sequence decoders (`\"`, `\\`, `\n`, `\r`, `\t`) in `main.cpp`.

5. **HNSW Edge Case & Entry Point Re-Selection**:
   - Fixed `HNSW::remove` to correctly re-assign `entryId_` to the node with the maximum remaining layer when the entry point is deleted.
   - Protected `randomLayer()` against `log(0)`.
   - Prevented self-loop edges in neighbor connections.

6. **Document Retrieval Thresholding**:
   - `DocumentDB::search` previously discarded results where distance > 0.7, causing valid semantic document matches to return empty hits.
   - Adjusted thresholding to return true top-$k$ nearest neighbors reliably.

7. **End-to-End Compatibility in Python Wrapper & UI**:
   - Enhanced `api_client.py` with `SearchHit` supporting both tuple-unpacking `(dist, id)` and attribute access (`hit.metadata`, `hit.category`).
   - Improved `app.py` with real-time status badges for both C++ server and Ollama, error handling, and visual expanders for stored items.

---

## 🚀 Quickstart Guide

### 1. Start Ollama (Embedding & LLM Models)
Ensure Ollama is running and has the required models installed:
```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 2. Launch the Entire Architecture
Simply run:
```cmd
start.bat
```
This automatically starts `db.exe` in the background and launches the Streamlit web application in your browser.

Alternatively, run manually in two terminals:

**Terminal 1 (Vector DB Server):**
```powershell
.\db.exe
```

**Terminal 2 (Streamlit UI):**
```powershell
streamlit run app.py
```

### 3. Rebuilding the C++ Server
To rebuild `db.exe` at any time:
```cmd
build.bat
```
*(Uses MSYS2 `g++` located at `C:\msys64\ucrt64\bin\g++.exe` with `-std=c++17 -O2 -static -lws2_32`)*

---

## 🧪 Automated Tests

Run the full automated test suite anytime to verify all components:

- **Vector Database Endpoints Test**:
  ```powershell
  python test_system.py
  ```
  *(Tests `/status`, `/items`, `/search` across HNSW/KDTree/BruteForce, `/insert`, `/delete`, `/doc/insert`, `/doc/search`, `/doc/delete`)*

- **RAG + Ollama End-to-End Test**:
  ```powershell
  python test_rag.py
  ```
  *(Tests text chunking, `nomic-embed-text` vector embedding, C++ storage, nearest-neighbor retrieval, and `llama3.2` question answering)*
