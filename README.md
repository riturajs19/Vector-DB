# C++ Vector Database & Local RAG System

A high-performance, in-memory Vector Database built from scratch in C++17, paired with a Local Retrieval-Augmented Generation (RAG) pipeline powered by Ollama and an interactive Streamlit web interface.

---

## Overview

Modern Gen AI applications rely heavily on vector databases for semantic search and context retrieval. Rather than using pre-built vector database libraries, this project implements a custom multithreaded vector search engine in C++17 alongside an end-to-end Python RAG workflow.

The system runs completely locally with zero external API costs, combining native C++ indexing speed with local Large Language Models.

---

## Core Capabilities

### 1. Custom C++ Vector Search Engine
- **Multiple Indexing Algorithms**: Implements **HNSW (Hierarchical Navigable Small World)**, **KD-Tree**, and **Brute-Force** search from scratch.
- **Distance Metrics**: Supports **Cosine Similarity**, **Euclidean ($L_2$)**, and **Manhattan ($L_1$)** distance calculations.
- **Adaptive Indexing**: Dynamically switches between exact Brute-Force search for small collections and approximate HNSW graph search for larger document stores.
- **Thread-Safe REST API**: Exposes concurrent HTTP endpoints on port `8080` using mutex-protected storage.

### 2. Local RAG Pipeline
- **Sliding-Window Chunking**: Splits long documents into 250-word segments with a 30-word overlap to preserve semantic continuity across boundaries.
- **Semantic Embeddings**: Converts text chunks and user queries into 768-dimensional dense vectors using Ollama (`nomic-embed-text`).
- **Context-Grounded Generation**: Retrieves the top-$k$ most relevant chunks from the C++ vector engine and generates accurate answers using `llama3.2`.

### 3. Interactive Streamlit Interface
- **Algorithm Benchmarking**: Insert custom vectors and compare retrieval results and distances across HNSW, KD-Tree, and Brute-Force side by side.
- **Document Management**: Ingest articles or notes, view stored chunks, and manage the document index in real time.
- **Explainable AI Q&A**: View generated answers alongside the exact source chunks and similarity distances retrieved from the database.

---

## Tech Stack

- **Database Engine**: C++17, STL, `cpp-httplib`
- **Backend & RAG**: Python 3.10+, `requests`
- **Frontend**: Streamlit
- **LLM & Embeddings**: Ollama (`llama3.2`, `nomic-embed-text`)

---

## Repository Structure

- `src/types.h` — Core vector and neighbor data structures
- `src/metrics.h` — Cosine, Euclidean, and Manhattan distance implementations
- `src/bruteforce.h` — Exact linear scan nearest-neighbor search
- `src/kdtree.h` — Balanced KD-Tree spatial partitioning index
- `src/hnsw.h` — Multi-layer HNSW proximity graph index
- `src/vectordb.h` — Thread-safe multi-algorithm vector store
- `src/documentdb.h` — Thread-safe document chunk store for RAG
- `main.cpp` — C++ HTTP REST server implementation
- `api_client.py` — Python client wrapper for the C++ REST API
- `ollama_client.py` — Python interface for local Ollama models
- `rag.py` — Text chunking, embedding, and retrieval-augmented generation logic
- `app.py` — Streamlit web application
- `build.bat` — Windows build script to compile `main.cpp` into `db.exe`
- `start.bat` — Startup script to launch the C++ server and Streamlit UI
- `test_system.py` — Automated integration tests for REST endpoints
- `test_rag.py` — End-to-end RAG verification script

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running locally
- MinGW-w64 / MSYS2 `g++` compiler *(optional, only needed to recompile `db.exe`)*

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Required Ollama Models
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 3. Launch the Application
Run the startup script to launch both the C++ server and the Streamlit interface:
```cmd
start.bat
```

Alternatively, run the services manually in two separate terminals:

**Terminal 1 — Start the C++ Vector Database Server:**
```bash
.\db.exe
```

**Terminal 2 — Start the Streamlit Web Interface:**
```bash
streamlit run app.py
```

### 4. Rebuild C++ Server from Source (Optional)
```cmd
build.bat
```

---

## API Reference (Port 8080)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/status` | Health check returning vector counts and dimensions |
| `GET` | `/items` | Retrieve all vectors in the benchmark store |
| `POST` | `/insert` | Insert a vector with metadata and category |
| `GET` | `/search` | Query top-$k$ nearest neighbors (`algo`: `hnsw`, `kdtree`, `bruteforce`) |
| `DELETE` | `/delete/:id` | Remove a vector by ID |
| `POST` | `/doc/insert` | Store a document chunk and its 768D embedding |
| `POST` | `/doc/search` | Retrieve top-$k$ matching chunks for a query embedding |
| `GET` | `/doc/list` | List all stored document chunks |
| `DELETE` | `/doc/delete/:id` | Delete a document chunk by ID |

---

## Testing

With the C++ server and Ollama running, execute the test suites to validate all endpoints and the RAG pipeline:

```bash
python test_system.py
python test_rag.py
```
