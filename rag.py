import ollama_client as oc
import api_client as vc


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = max(1, chunk_words - overlap_words)
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


def insert_document(doc_db, title: str, text: str) -> list[int]:
    chunks = chunk_text(text)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = oc.embed(chunk)
        chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))
    return ids


def search(doc_db, question: str, k: int = 3):
    q_emb = oc.embed(question)
    return doc_db.search(q_emb, k)


def ask(doc_db, question: str, k: int = 3) -> dict:
    q_emb = oc.embed(question)
    hits = doc_db.search(q_emb, k)

    context = ""
    for i, (dist, doc) in enumerate(hits):
        context += f"[{i+1}] {doc.title}:\n{doc.text}\n\n"

    if context:
        prompt = (
            "You are a helpful assistant. Answer the user's question directly using the provided context.\n\n"
            f"Context:\n{context}\n"
            f"Question: {question}\n\nAnswer:"
        )
    else:
        prompt = (
            "You are a helpful assistant. Answer the user's question directly.\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    answer = oc.generate(prompt)
    return {
        "answer": answer,
        "contexts": [{"title": d.title, "text": d.text, "distance": dist} for dist, d in hits],
    }
