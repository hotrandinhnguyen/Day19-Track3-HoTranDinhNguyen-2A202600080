"""
Buoc 3: Xay dung do thi tri thuc bang NetworkX tu triples.json
Output: data/graph.gpickle + anh visualization
"""

import json
import os
import sys
import pickle
import time
from collections import defaultdict

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from openai import OpenAI
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── 1. LOAD TRIPLES ──────────────────────────────────────────────────────────

def load_triples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 2. DEDUPLICATION ─────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Chuan hoa ten entity de khử trung lap."""
    return text.strip().lower().rstrip(".")


def deduplicate_triples(triples: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for t in triples:
        key = (normalize(t["s"]), t["p"], normalize(t["o"]))
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


# ── 3. BUILD GRAPH ────────────────────────────────────────────────────────────

def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    """
    MultiDiGraph: co huong, cho phep nhieu canh giua 2 node
    (vi du: A FOUNDED_BY B va A PARTNER_OF B co the cung ton tai)
    """
    G = nx.MultiDiGraph()

    for t in triples:
        s = t["s"].strip()
        o = t["o"].strip()
        p = t["p"]
        source_title = t.get("source_title", "")

        # Them node neu chua co
        if not G.has_node(s):
            G.add_node(s, entity_type="unknown")
        if not G.has_node(o):
            G.add_node(o, entity_type="unknown")

        # Them canh co nhan la predicate
        G.add_edge(s, o, relation=p, source=source_title)

    # Gan entity_type dua vao predicate
    for s, o, data in G.edges(data=True):
        rel = data["relation"]
        if rel in ("FOUNDED_BY", "CEO_OF", "EMPLOYEE_OF"):
            G.nodes[o]["entity_type"] = "person"
        elif rel == "FOUNDED_IN":
            G.nodes[o]["entity_type"] = "year"
        elif rel == "LOCATED_IN":
            G.nodes[o]["entity_type"] = "location"
        elif rel in ("PRODUCT_OF",):
            G.nodes[o]["entity_type"] = "product"
        else:
            if G.nodes[s]["entity_type"] == "unknown":
                G.nodes[s]["entity_type"] = "organization"

    return G


# ── 4. STATS ──────────────────────────────────────────────────────────────────

def print_stats(graph: nx.MultiDiGraph, triples_raw: int, triples_dedup: int):
    sep = "=" * 50
    print(f"\n{sep}")
    print("DO THI TRI THUC — THONG KE")
    print(sep)
    print(f"  Triples goc:        {triples_raw:>6,}")
    print(f"  Triples sau dedup:  {triples_dedup:>6,}  (bo {triples_raw - triples_dedup:,} trung lap)")
    print(f"  Nodes:              {graph.number_of_nodes():>6,}")
    print(f"  Edges:              {graph.number_of_edges():>6,}")
    print(f"  Density:            {nx.density(graph):.6f}")

    degree_seq = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 node nhieu ket noi nhat:")
    for node, deg in degree_seq[:10]:
        print(f"    {node:35s}  degree={deg}")


# ── 5. VISUALIZATION ──────────────────────────────────────────────────────────

COLOR_MAP = {
    "organization": "#4C9BE8",
    "person":       "#E8834C",
    "year":         "#A8D8A8",
    "location":     "#E8D84C",
    "product":      "#C84CE8",
    "unknown":      "#AAAAAA",
}

def visualize_subgraph(graph: nx.MultiDiGraph, center: str = "OpenAI",
                       radius: int = 2, output_path: str = "data/graph_openai.png"):
    """Visualize subgraph xung quanh mot node trung tam."""
    if center not in graph:
        print(f"  [WARN] Node '{center}' khong ton tai trong do thi")
        return

    sub_nodes = nx.single_source_shortest_path_length(graph, center, cutoff=radius).keys()
    sg = graph.subgraph(sub_nodes).copy()

    _, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(sg, seed=42, k=2.5)

    node_colors = [COLOR_MAP.get(sg.nodes[n].get("entity_type", "unknown"), "#AAAAAA")
                   for n in sg.nodes()]
    node_sizes  = [800 if n == center else 400 for n in sg.nodes()]

    nx.draw_networkx_nodes(sg, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(sg, pos, font_size=7, font_weight="bold", ax=ax)

    # Ve canh voi nhan relation
    edge_labels = {(u, v): d["relation"] for u, v, d in sg.edges(data=True)}
    nx.draw_networkx_edges(sg, pos, arrows=True, arrowsize=15,
                           edge_color="#666666", alpha=0.6,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_edge_labels(sg, pos, edge_labels=edge_labels,
                                 font_size=5, alpha=0.8, ax=ax)

    # Legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in COLOR_MAP.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=8)

    ax.set_title(f"Knowledge Graph — Subgraph xung quanh '{center}' (2-hop)\n"
                 f"{sg.number_of_nodes()} nodes, {sg.number_of_edges()} edges",
                 fontsize=13, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_full_overview(graph: nx.MultiDiGraph, output_path: str = "data/graph_overview.png"):
    """Visualize toan bo do thi (chi hien node nhieu ket noi)."""
    top_nodes = [n for n, _ in sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:80]]
    sg = graph.subgraph(top_nodes).copy()

    _, ax = plt.subplots(figsize=(20, 16))
    pos = nx.spring_layout(sg, seed=0, k=1.8)

    node_colors = [COLOR_MAP.get(sg.nodes[n].get("entity_type", "unknown"), "#AAAAAA")
                   for n in sg.nodes()]
    degrees = dict(sg.degree())
    node_sizes = [200 + degrees[n] * 30 for n in sg.nodes()]

    nx.draw_networkx_nodes(sg, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(sg, pos, font_size=6, ax=ax)
    nx.draw_networkx_edges(sg, pos, arrows=True, arrowsize=8,
                           edge_color="#999999", alpha=0.4, ax=ax)

    patches = [mpatches.Patch(color=c, label=t) for t, c in COLOR_MAP.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    ax.set_title(
        f"Knowledge Graph Overview — Top 80 nodes by degree\n"
        f"Total graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges",
        fontsize=14, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── 6. NODE EMBEDDINGS ───────────────────────────────────────────────────────

def add_node_embeddings(graph: nx.MultiDiGraph, batch_size: int = 100) -> None:
    """
    Them embedding vector cho moi node (dung text-embedding-3-small).
    Luu vao node attribute 'embedding'.
    """
    nodes = list(graph.nodes())
    print(f"   Embedding {len(nodes)} nodes...")

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        for node, emb_obj in zip(batch, resp.data):
            graph.nodes[node]["embedding"] = emb_obj.embedding

        done = min(i + batch_size, len(nodes))
        print(f"   {done}/{len(nodes)} nodes embedded")
        time.sleep(0.3)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    print("1. Loading triples...")
    triples_raw = load_triples("data/triples_v2.json")
    print(f"   {len(triples_raw)} triples loaded")

    print("\n2. Deduplication...")
    triples = deduplicate_triples(triples_raw)
    print(f"   {len(triples)} triples sau dedup")

    print("\n3. Building graph...")
    graph = build_graph(triples)

    print_stats(graph, len(triples_raw), len(triples))

    print("\n4. Adding node embeddings...")
    add_node_embeddings(graph)

    print("\n5. Saving graph...")
    with open("data/graph.gpickle", "wb") as f:
        pickle.dump(graph, f)
    print("   Saved: data/graph.gpickle")

    print("\n6. Visualizing...")
    visualize_subgraph(graph, center="OpenAI",          radius=2, output_path="data/graph_openai.png")
    visualize_subgraph(graph, center="Google DeepMind", radius=2, output_path="data/graph_deepmind.png")
    visualize_full_overview(graph, output_path="data/graph_overview.png")

    print("\nBuoc 3 hoan thanh!")
    print("  data/graph.gpickle      — do thi + embeddings")
    print("  data/graph_openai.png   — visualization OpenAI subgraph")
    print("  data/graph_deepmind.png — visualization DeepMind subgraph")
    print("  data/graph_overview.png — toan canh do thi")


if __name__ == "__main__":
    main()
