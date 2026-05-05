"""
Buoc 4: GraphRAG retrieval pipeline
Query -> Seed nodes -> BFS traversal -> Subgraph-to-text -> LLM answer
"""

import json
import os
import sys
import pickle
import time
import math
from openai import OpenAI
from dotenv import load_dotenv
import networkx as nx

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── 1. LOAD GRAPH ─────────────────────────────────────────────────────────────

def load_graph(path: str = "data/graph.gpickle") -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── 2. EXTRACT ENTITIES FROM QUERY ───────────────────────────────────────────

def extract_query_entities(query: str) -> list[str]:
    """Dung LLM trich xuat ten thuc the chinh trong cau hoi."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Extract named entities (companies, people, products, organizations) "
                "from the query. Return a JSON array of strings only. "
                "Example: [\"OpenAI\", \"Sam Altman\"]"
            )},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=100,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except Exception:
        return []


# ── 3. FIND SEED NODES ────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def find_seed_nodes(entities: list[str], graph: nx.MultiDiGraph) -> list[str]:
    """
    Tim node trong do thi khop voi entity trong query.
    Luot 1: khop chinh xac/phan (nhanh, khong ton API).
    Luot 2: cosine similarity voi node embeddings (neu co).
    """
    all_nodes = list(graph.nodes())
    seeds = []

    for entity in entities:
        entity_lower = entity.lower()

        # Luot 1a: khop chinh xac
        exact = [n for n in all_nodes if n.lower() == entity_lower]
        if exact:
            seeds.extend(exact)
            continue

        # Luot 1b: khop phan
        partial = [n for n in all_nodes
                   if entity_lower in n.lower() or n.lower() in entity_lower]
        if partial:
            seeds.extend(partial[:3])
            continue

        # Luot 1c: khop tung tu
        words = [w for w in entity_lower.split() if len(w) > 2]
        word_match = [n for n in all_nodes
                      if any(w in n.lower() for w in words)]
        if word_match:
            seeds.extend(word_match[:3])
            continue

        # Luot 2: cosine similarity voi node embeddings
        nodes_with_emb = [n for n in all_nodes
                          if graph.nodes[n].get("embedding")]
        if not nodes_with_emb:
            continue

        resp = client.embeddings.create(
            model="text-embedding-3-small", input=[entity]
        )
        q_emb = resp.data[0].embedding
        scored = [
            (n, _cosine(q_emb, graph.nodes[n]["embedding"]))
            for n in nodes_with_emb
        ]
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:3]
        seeds.extend(n for n, score in top if score > 0.6)

    return list(dict.fromkeys(seeds))


def load_corpus(path: str = "data/corpus.json") -> dict[str, str]:
    """Load corpus de fallback khi graph context rong."""
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    return {a["title"].lower(): a["text"] for a in articles}


# ── 4. BFS SUBGRAPH ───────────────────────────────────────────────────────────

def get_bfs_subgraph(graph: nx.MultiDiGraph, seed_nodes: list[str],
                     depth: int = 2) -> nx.MultiDiGraph:
    """Lay subgraph 2-hop tu cac seed node."""
    visited = set()
    for seed in seed_nodes:
        if seed not in graph:
            continue
        reachable = nx.single_source_shortest_path_length(
            graph.to_undirected(), seed, cutoff=depth
        )
        visited.update(reachable.keys())
    if not visited:
        return nx.MultiDiGraph()
    return graph.subgraph(visited).copy()


# ── 5. SUBGRAPH TO TEXT ───────────────────────────────────────────────────────

def subgraph_to_text(subgraph: nx.MultiDiGraph, seed_nodes: list[str] | None = None,
                     max_triples: int = 60) -> str:
    """
    Chuyen subgraph thanh doan van de gua cho LLM.
    Uu tien: triples lien quan truc tiep den seed nodes truoc,
    sau do cac triples xa hon.
    """
    seed_set = set(seed_nodes or [])
    direct, indirect = [], []

    for s, o, data in subgraph.edges(data=True):
        rel = data.get("relation", "RELATED_TO")
        src = data.get("source", "")
        suffix = f"  [src: {src}]" if src else ""
        line = f"- {s} {rel} {o}{suffix}"
        if s in seed_set or o in seed_set:
            direct.append(line)
        else:
            indirect.append(line)

    lines = direct + indirect
    return "\n".join(lines[:max_triples])


# ── 6. LLM ANSWER ─────────────────────────────────────────────────────────────

def llm_answer(query: str, context: str, system: str = "graphrag") -> tuple[str, int]:
    """Goi LLM voi context tu do thi, tra ve (answer, total_tokens)."""
    if system == "graphrag":
        sys_msg = (
            "You are a helpful AI assistant. Answer the question using ONLY "
            "the knowledge graph facts provided below. Be concise and factual.\n\n"
            f"Knowledge Graph Facts:\n{context}"
        )
    else:
        sys_msg = (
            "You are a helpful AI assistant. Answer the question based on your "
            "general knowledge about AI companies and technology."
        )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=300,
    )
    answer = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens
    return answer, tokens


# ── 7. GRAPHRAG PIPELINE ──────────────────────────────────────────────────────

_corpus_cache: dict[str, str] | None = None


def _corpus_fallback(entities: list[str]) -> str:
    """Tra ve doan text corpus khi graph context rong."""
    global _corpus_cache
    if _corpus_cache is None:
        _corpus_cache = load_corpus()
    texts = []
    for entity in entities:
        entity_lower = entity.lower()
        for title_lower, text in _corpus_cache.items():
            if entity_lower in title_lower or title_lower in entity_lower:
                texts.append(f"[From article '{title_lower}']:\n{text[:800]}")
                break
    return "\n\n".join(texts)


def graphrag_query(query: str, graph: nx.MultiDiGraph,
                   verbose: bool = True, depth: int = 2) -> dict:
    """Pipeline day du: query -> answer."""
    t0 = time.time()

    # Step 1: Extract entities
    entities = extract_query_entities(query)

    # Step 2: Find seed nodes
    seeds = find_seed_nodes(entities, graph)

    # Step 3: BFS subgraph
    sg = get_bfs_subgraph(graph, seeds, depth=depth)

    # Step 4: Subgraph to text — seed-adjacent triples first
    context = subgraph_to_text(sg, seed_nodes=seeds, max_triples=60)

    # Fallback: neu graph context rong, dung doan text tu corpus
    if not context:
        context = _corpus_fallback(entities)

    # Step 5: LLM answer
    if context:
        answer, tokens = llm_answer(query, context, system="graphrag")
    else:
        answer = "Khong tim thay thong tin lien quan."
        tokens = 0

    latency = round(time.time() - t0, 2)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Query:    {query}")
        print(f"Entities: {entities}")
        print(f"Seeds:    {seeds}")
        print(f"Subgraph: {sg.number_of_nodes()} nodes, {sg.number_of_edges()} edges")
        print(f"Answer:   {answer}")
        print(f"Tokens:   {tokens} | Latency: {latency}s")

    return {
        "query": query,
        "entities": entities,
        "seeds": seeds,
        "subgraph_nodes": sg.number_of_nodes(),
        "subgraph_edges": sg.number_of_edges(),
        "context_triples": len(context.splitlines()),
        "answer": answer,
        "tokens": tokens,
        "latency": latency,
    }


# ── DEMO ──────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "What is OpenAI?",
    "Which AI companies were co-founded by former Google employees?",
    "What products has Microsoft invested in or acquired in the AI space?",
]

def main():
    print("Loading graph...")
    graph = load_graph()
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")

    print("=" * 60)
    print("DEMO: GraphRAG vs Flat RAG")
    print("=" * 60)

    for q in DEMO_QUERIES:
        print(f"\n{'='*60}")
        print("[GraphRAG]")
        graphrag_query(q, graph, verbose=True)

    print("\n\nPipeline san sang. Chay step5_benchmark.py de benchmark day du.")


if __name__ == "__main__":
    main()
