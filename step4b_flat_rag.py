"""
Flat RAG dung: ChromaDB vector search tren corpus 100 bai
(khong dung general knowledge cua LLM)
"""

import json
import os
import sys
import time
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "tech_corpus"


# ── BUILD VECTOR STORE ────────────────────────────────────────────────────────

def build_vectorstore(corpus_path: str = "data/corpus.json") -> chromadb.Collection:
    """Index corpus vao ChromaDB (chi chay lan dau)."""
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    # Neu collection da ton tai, dung lai
    try:
        col = chroma.get_collection(COLLECTION_NAME)
        print(f"ChromaDB da co {col.count()} chunks, dung lai.")
        return col
    except Exception:
        pass

    print("Dang index corpus vao ChromaDB...")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    col = chroma.create_collection(COLLECTION_NAME, embedding_function=openai_ef)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Chia moi bai thanh 2 chunks (dau va cuoi) de co nhieu retrieval points
    ids, docs, metas = [], [], []
    for art in corpus:
        text = art["text"]
        mid = len(text) // 2
        chunks = [text[:mid], text[mid:]] if len(text) > 500 else [text]
        for i, chunk in enumerate(chunks):
            ids.append(f"{art['id']}_chunk{i}")
            docs.append(chunk)
            metas.append({"title": art["title"], "url": art["url"]})

    # Batch insert 50 chunks mot lan
    batch = 50
    for i in range(0, len(ids), batch):
        col.add(ids=ids[i:i+batch], documents=docs[i:i+batch], metadatas=metas[i:i+batch])
        print(f"  Indexed {min(i+batch, len(ids))}/{len(ids)} chunks")

    print(f"Hoan thanh: {col.count()} chunks trong ChromaDB")
    return col


# ── FLAT RAG QUERY ────────────────────────────────────────────────────────────

def flat_rag_query(query: str, col: chromadb.Collection,
                   top_k: int = 5, verbose: bool = True) -> dict:
    """
    Flat RAG thuc su:
    1. Vector search top-k chunks tu corpus
    2. Gop context → LLM answer
    """
    t0 = time.time()

    # Retrieve
    results = col.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Build context
    context_parts = []
    for doc, meta in zip(docs, metas):
        context_parts.append(f"[Source: {meta['title']}]\n{doc[:400]}")
    context = "\n\n---\n\n".join(context_parts)

    # LLM answer — chi dung corpus, khong dung general knowledge
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant. Answer the question using ONLY "
                "the provided source documents. If the answer is not in the documents, "
                "say 'Not found in corpus'.\n\n"
                f"Documents:\n{context}"
            )},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=300,
    )
    answer = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens
    latency = round(time.time() - t0, 2)

    sources = [m["title"] for m in metas]

    if verbose:
        print(f"\n[Flat RAG] {query}")
        print(f"Sources: {sources}")
        print(f"Answer:  {answer[:200]}")
        print(f"Tokens:  {tokens} | Latency: {latency}s")

    return {
        "query": query,
        "answer": answer,
        "tokens": tokens,
        "latency": latency,
        "sources": sources,
    }


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    col = build_vectorstore()

    test_queries = [
        "Who founded OpenAI?",
        "Which AI companies were co-founded by former Google employees?",
    ]
    for q in test_queries:
        flat_rag_query(q, col, verbose=True)
