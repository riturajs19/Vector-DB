
# ⚡ C++ Vector Database & Local RAG System

A custom in-memory **Vector Database** built from scratch in **C++17** (implementing **HNSW**, **KD-Tree**, and **Brute-Force** nearest-neighbor search), integrated with a **Local RAG Pipeline** powered by **Ollama** and an interactive **Streamlit** dashboard.

---

## 📌 Overview

Most RAG applications rely on black-box vector database libraries. This project builds both the **high-performance C++ vector indexing engine** and the **Python Retrieval-Augmented Generation (RAG) orchestration layer** from scratch:

- **C++17 Vector Engine (`db.exe`)**: Exposes a thread-safe HTTP REST API for low-latency vector indexing and nearest-neighbor retrieval.
- **Local AI & Embeddings (Ollama)**: Generates 768-dimensional semantic embeddings using `nomic-embed-text` and grounded answers using `llama3.2`—100% locally with zero API cost.
- **Interactive Web UI (Streamlit)**: Provides live health monitoring, algorithm benchmarking, document ingestion, and context-backed Q&A.

---

## ✨ Key Features

- **Three Search Algorithms from Scratch**:
  - **HNSW (Hierarchical Navigable Small World)**: Multi-layer proximity graph for fast $O(\log N)$ approximate nearest-neighbor search in high dimensions.
  - **KD-Tree**: Balanced spatial partitioning tree for exact $O(\log N)$ search in low dimensions.
  - **Brute-Force**: Exact $O(N)$ linear scan baseline.
- **Multiple Distance Metrics**: Supports **Cosine**, **Euclidean ($L_2$)**, and **Manhattan ($L_1$)** similarity metrics.
- **Hybrid RAG Storage (`DocumentDB`)**: Automatically switches between Brute-Force (for small collections) and HNSW (for larger document collections).
- **Sliding-Window Document Chunking**: Splits long documents into 250-word chunks with 30-word overlap to preserve context across boundaries.
- **Thread-Safe REST API**: Multithreaded C++ HTTP server protected with mutex locking (`std::lock_guard`).

---

## 🏛️ System Architecture

┌─────────────────────────────────────────────────────────────────┐
│               Streamlit Web UI & RAG Layer (Python)             │
│         (Demo Search  |  Document Ingestion  |  Ask AI)         │
└───────────────┬─────────────────────────────────┬───────────────┘
                │ HTTP REST (Port 8080)           │ HTTP REST (Port 11434)
                ▼                                 ▼
┌───────────────────────────────────┐ ┌───────────────────────────┐
│   C++ VectorDB Server (db.exe)    │ │    Local Ollama Service   │
│  • HNSW, KD-Tree & Brute-Force    │ │  • nomic-embed-text (768D)│
│  • VectorDB (4D Benchmark Store)  │ │  • llama3.2 (Local LLM)   │
│  • DocumentDB (768D RAG Store)    │ └───────────────────────────┘
└───────────────────────────────────┘

---

## 📂 Project Structure

├── src/
│   ├── types.h          # Core vector and neighbor data structures
│   ├── metrics.h        # Cosine, Euclidean, and Manhattan distance functions
│   ├── bruteforce.h     # Exact linear scan search index
│   ├── kdtree.h         # Balanced KD-Tree spatial index
│   ├── hnsw.h           # Multi-layer HNSW graph index
│   ├── vectordb.h       # Thread-safe multi-algorithm vector database
│   └── documentdb.h     # Thread-safe 768D RAG document chunk database
├── main.cpp             # C++ HTTP REST server (Port 8080)
├── httplib.h            # Single-header C++ HTTP server library
├── api_client.py        # Python REST client for the C++ VectorDB server
├── ollama_client.py     # Python client for Ollama embeddings & LLM generation
├── rag.py               # Document chunking, embedding, and RAG query pipeline
├── app.py               # Streamlit web application
├── build.bat            # Script to compile main.cpp into db.exe
├── start.bat            # One-click launcher for C++ server + Streamlit UI
├── test_system.py       # Automated REST API integration tests
├── test_rag.py          # Automated end-to-end RAG pipeline tests
└── requirements.txt     # Python dependencies

---

## 🛠️ Tech Stack

- **Core Database Engine**: C++17, STL (`std::priority_queue`, `std::unordered_map`, `std::mutex`), `cpp-httplib`
- **RAG & Backend**: Python 3.10+, `requests`
- **Frontend UI**: Streamlit
- **Local AI Models**: Ollama (`nomic-embed-text` for embeddings, `llama3.2` for LLM Q&A)

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- **Ollama** installed and running (https://ollama.com)
- **G++ Compiler (MinGW-w64 / MSYS2)** *(only required if rebuilding `db.exe` from source)*

### 2. Install Python Dependencies
pip install -r requirements.txt

### 3. Pull Required Ollama Models
ollama pull nomic-embed-text
ollama pull llama3.2

### 4. Run the Application
Launch both the C++ Vector Database server and the Streamlit UI with one command:
start.bat

Or start them manually in two terminals:
# Terminal 1: Start C++ VectorDB Server (http://localhost:8080)
.\db.exe

# Terminal 2: Start Streamlit Web UI (http://localhost:8501)
streamlit run app.py

---

## 🔌 REST API Endpoints (C++ Server — Port 8080)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/status` | Returns vector counts and dimensions for both databases |
| `GET` | `/items` | Lists all stored vectors in the 4D benchmark database |
| `POST` | `/insert` | Inserts a new vector with metadata and category |
| `GET` | `/search` | Queries top-k nearest neighbors using `hnsw`, `kdtree`, or `bruteforce` |
| `DELETE` | `/delete/:id` | Deletes a vector by ID from the benchmark database |
| `POST` | `/doc/insert` | Inserts a document chunk and its 768D embedding |
| `POST` | `/doc/search` | Retrieves top-k most relevant document chunks for RAG |
| `GET` | `/doc/list` | Lists all stored document chunks |
| `DELETE` | `/doc/delete/:id` | Deletes a document chunk by ID |

---

## 🧪 Running Tests

# Test all C++ VectorDB REST endpoints
python test_system.py

# Test end-to-end document chunking, embedding, retrieval, and LLM answering
python test_rag.py
