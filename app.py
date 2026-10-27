import streamlit as st
import ollama_client as oc
import api_client as vc
import rag
import requests

API = "http://localhost:8080"

st.set_page_config(page_title="Vector Database & RAG", layout="wide", page_icon="⚡")

# Check C++ server connection
def check_vdb_server():
    try:
        r = requests.get(f"{API}/status", timeout=1)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}

vdb_ok, vdb_status = check_vdb_server()
ollama_ok = oc.is_available()

# Sidebar: System Status & Controls
with st.sidebar:
    st.header("⚡ System Status")
    st.write("VectorDB Server (C++):", "🟢 Online" if vdb_ok else "🔴 Offline (`localhost:8080`)")
    if vdb_ok and vdb_status:
        st.caption(f"Demo vectors: {vdb_status.get('demoCount', 0)} ({vdb_status.get('demoDims', 0)}D)")
        st.caption(f"Doc chunks: {vdb_status.get('docCount', 0)} ({vdb_status.get('docDims', 0)}D)")

    st.write("Ollama Service:", "🟢 Online" if ollama_ok else "🔴 Offline (`localhost:11434`)")
    if ollama_ok:
        st.caption("Embedding model: `nomic-embed-text`")
        st.caption("LLM: `llama3.2`")

    st.divider()
    if st.button("🔄 Refresh Status"):
        st.rerun()

if "doc_db" not in st.session_state:
    st.session_state.doc_db = vc.DocumentDB()
if "demo_db" not in st.session_state:
    st.session_state.demo_db = vc.VectorDB()

st.title("⚡ Vector Database — HNSW + KD-Tree + RAG")

if not vdb_ok:
    st.error("⚠️ The C++ Vector Database server is not reachable at http://localhost:8080. Please ensure the server (`db.exe`) is running.")

tab_search, tab_docs, tab_ask = st.tabs(["Demo Search", "Documents", "Ask AI"])

# =====================================================================
# Tab 1: Demo Search
# =====================================================================
with tab_search:
    st.subheader("Compare algorithms on demo vectors")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ➕ Insert Custom Vector")
        meta = st.text_input("Description", placeholder="e.g. Quantitative Risk Analysis")
        cat = st.selectbox("Category", ["hr", "finance", "other"])
        emb_str = st.text_input("Embedding (4 comma-separated floats)", "0.9,0.1,0.87,0.1")
        if st.button("Insert Vector"):
            if not vdb_ok:
                st.error("VectorDB server is offline.")
            elif not meta.strip():
                st.warning("Please provide a description.")
            else:
                try:
                    emb = [float(x.strip()) for x in emb_str.split(",") if x.strip()]
                    new_id = st.session_state.demo_db.insert(meta, cat, emb)
                    st.success(f"Inserted item id={new_id}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error inserting: {e}")

    with col2:
        st.markdown("#### 🔍 Query Nearest Neighbors")
        q_str = st.text_input("Query embedding", "0.86,0.06,0.09,0.88")
        k = st.slider("k (number of neighbors)", 1, 10, 3)
        algo = st.selectbox("Algorithm", ["hnsw", "kdtree", "bruteforce"])
        if st.button("Search Demo"):
            if not vdb_ok:
                st.error("VectorDB server is offline.")
            else:
                try:
                    q = [float(x.strip()) for x in q_str.split(",") if x.strip()]
                    results = st.session_state.demo_db.search(q, k, algo)
                    if not results:
                        st.info("No results found.")
                    else:
                        st.markdown(f"**Results ({algo.upper()}):**")
                        for hit in results:
                            st.write(f"- **id={hit.id}** | distance={hit.distance:.4f} | **[{hit.category}]** {hit.metadata}")
                except Exception as e:
                    st.error(f"Search failed: {e}")

    st.divider()
    demo_count = st.session_state.demo_db.size() if vdb_ok else 0
    st.write(f"**Total demo vectors in DB:** {demo_count}")
    if vdb_ok and demo_count > 0:
        with st.expander("View all demo vectors in DB"):
            try:
                items = st.session_state.demo_db.all()
                for item in items:
                    c_del, c_info = st.columns([1, 8])
                    with c_del:
                        if st.button("Delete", key=f"demo_del_{item['id']}"):
                            st.session_state.demo_db.remove(item["id"])
                            st.rerun()
                    with c_info:
                        st.write(f"**id={item['id']}** `[{item['category']}]` {item['metadata']}")
            except Exception as e:
                st.error(f"Could not load items: {e}")

# =====================================================================
# Tab 2: Documents
# =====================================================================
with tab_docs:
    st.subheader("Document Ingestion & Chunking")

    title = st.text_input("Document title", placeholder="e.g. Distributed Computing Fundamentals")
    text = st.text_area("Document text", height=200, placeholder="Paste your article, documentation, or notes here...")

    if st.button("Insert Document"):
        if not vdb_ok:
            st.error("VectorDB server is offline.")
        elif not ollama_ok:
            st.error("Ollama service is offline. Please start Ollama to generate embeddings.")
        elif not title.strip() or not text.strip():
            st.warning("Please provide both document title and text.")
        else:
            with st.spinner("Chunking text and generating embeddings via Ollama..."):
                try:
                    ids = rag.insert_document(st.session_state.doc_db, title, text)
                    st.success(f"Successfully inserted {len(ids)} chunk(s) (IDs: {ids})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to ingest document: {e}")

    st.divider()
    doc_count = st.session_state.doc_db.size() if vdb_ok else 0
    st.write(f"**Total chunks stored in DocumentDB:** {doc_count}")

    if vdb_ok and doc_count > 0:
        try:
            docs = st.session_state.doc_db.all()
            for doc in docs:
                with st.expander(f"📄 {doc.title} (id={doc.id})"):
                    st.write(doc.text)
                    if st.button("Delete Chunk", key=f"del_{doc.id}"):
                        st.session_state.doc_db.remove(doc.id)
                        st.rerun()
        except Exception as e:
            st.error(f"Could not load documents: {e}")

# =====================================================================
# Tab 3: Ask AI (RAG)
# =====================================================================
with tab_ask:
    st.subheader("RAG: Semantic Search + Local LLM Q&A")
    question = st.text_input("Ask a question about your documents", placeholder="e.g. What is the difference between BFS and DFS?")
    k_chunks = st.slider("Context chunks to retrieve", 1, 5, 3)

    if st.button("Ask AI"):
        if not vdb_ok:
            st.error("VectorDB server is offline.")
        elif not ollama_ok:
            st.error("Ollama service is offline.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching vector index and prompting local LLM..."):
                try:
                    result = rag.ask(st.session_state.doc_db, question, k_chunks)
                    st.write("### 💬 Answer")
                    st.write(result["answer"])

                    st.write("### 📚 Retrieved Context Sources")
                    if result["contexts"]:
                        for i, ctx in enumerate(result["contexts"]):
                            with st.expander(f"[{i+1}] {ctx['title']} — distance: {ctx['distance']:.4f}"):
                                st.write(ctx["text"])
                    else:
                        st.info("No relevant chunks retrieved from the document database.")
                except Exception as e:
                    st.error(f"RAG query failed: {e}")
