"""
Buoc 5: Benchmark GraphRAG vs Flat RAG tren 20 cau hoi
- Do accuracy (LLM-as-judge), latency, token cost
- Xuat bang so sanh va phan tich failure modes
"""

import json
import os
import sys
import pickle
import time
import csv
from openai import OpenAI
from dotenv import load_dotenv

# Import pipeline tu buoc 4
sys.path.insert(0, os.path.dirname(__file__))
from step4_graphrag import graphrag_query, load_graph
from step4b_flat_rag import flat_rag_query, build_vectorstore

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── LLM-AS-JUDGE ─────────────────────────────────────────────────────────────

def judge_answer(question: str, ground_truth: str,
                 key_facts: list[str], answer: str) -> tuple[int, str]:
    """
    Dung LLM danh gia answer co dung khong.
    Tra ve (score 0/1, reasoning).
    """
    prompt = f"""You are an objective evaluator. Score the answer 1 (correct) or 0 (incorrect/hallucination).

Question: {question}

Ground Truth: {ground_truth}

Key Facts that must be present: {", ".join(key_facts)}

Answer to evaluate: {answer}

Rules:
- Score 1 if the answer contains most key facts and is factually correct
- Score 0 if the answer contains hallucinations, wrong facts, or misses most key facts
- Be strict about factual accuracy

Respond in JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        data = json.loads(raw)
        return int(data.get("score", 0)), data.get("reason", "")
    except Exception:
        return 0, "parse error"


# ── RUN BENCHMARK ─────────────────────────────────────────────────────────────

def run_benchmark(questions: list[dict], graph, col) -> list[dict]:
    results = []
    total = len(questions)

    print(f"\nBat dau benchmark {total} cau hoi...\n")
    print(f"{'ID':>3} {'Type':>10}  {'GraphRAG':>8}  {'FlatRAG':>8}  {'Latency GR':>10}  {'Latency FR':>10}")
    print("─" * 70)

    for q in questions:
        qid   = q["id"]
        qtype = q["type"]
        question    = q["question"]
        ground_truth = q["ground_truth"]
        key_facts   = q["key_facts"]

        # --- GraphRAG ---
        gr = graphrag_query(question, graph, verbose=False)
        gr_score, gr_reason = judge_answer(question, ground_truth, key_facts, gr["answer"])
        time.sleep(0.3)

        # --- Flat RAG (ChromaDB) ---
        fr = flat_rag_query(question, col, verbose=False)
        fr_score, fr_reason = judge_answer(question, ground_truth, key_facts, fr["answer"])
        time.sleep(0.3)

        row = {
            "id":             qid,
            "type":           qtype,
            "question":       question,
            "ground_truth":   ground_truth,
            # GraphRAG
            "gr_answer":      gr["answer"],
            "gr_score":       gr_score,
            "gr_reason":      gr_reason,
            "gr_tokens":      gr["tokens"],
            "gr_latency":     gr["latency"],
            "gr_nodes":       gr["subgraph_nodes"],
            "gr_edges":       gr["subgraph_edges"],
            # Flat RAG
            "fr_answer":      fr["answer"],
            "fr_score":       fr_score,
            "fr_reason":      fr_reason,
            "fr_tokens":      fr["tokens"],
            "fr_latency":     fr["latency"],
            # Failure mode
            "flat_hallucinated": (fr_score == 0 and gr_score == 1),
        }
        results.append(row)

        flag = " << GraphRAG wins" if row["flat_hallucinated"] else ""
        print(f"{qid:>3} {qtype:>10}  {gr_score:>8}  {fr_score:>8}  "
              f"{gr['latency']:>9.2f}s  {fr['latency']:>9.2f}s{flag}")

    return results


# ── SAVE RESULTS ──────────────────────────────────────────────────────────────

def save_csv(results: list[dict], path: str = "data/benchmark_results.csv"):
    fields = [
        "id", "type", "question",
        "gr_score", "fr_score", "flat_hallucinated",
        "gr_latency", "fr_latency",
        "gr_tokens", "fr_tokens",
        "gr_nodes", "gr_edges",
        "gr_answer", "fr_answer",
        "gr_reason", "fr_reason",
        "ground_truth",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"Saved: {path}")


# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    total = len(results)
    simple  = [r for r in results if r["type"] == "simple"]
    multi   = [r for r in results if r["type"] == "multi-hop"]
    halluc  = [r for r in results if r["flat_hallucinated"]]

    gr_acc_total = sum(r["gr_score"] for r in results) / total * 100
    fr_acc_total = sum(r["fr_score"] for r in results) / total * 100

    gr_acc_simple = sum(r["gr_score"] for r in simple) / len(simple) * 100 if simple else 0
    fr_acc_simple = sum(r["fr_score"] for r in simple) / len(simple) * 100 if simple else 0

    gr_acc_multi  = sum(r["gr_score"] for r in multi)  / len(multi)  * 100 if multi else 0
    fr_acc_multi  = sum(r["fr_score"] for r in multi)  / len(multi)  * 100 if multi else 0

    avg_gr_latency = sum(r["gr_latency"] for r in results) / total
    avg_fr_latency = sum(r["fr_latency"] for r in results) / total
    avg_gr_tokens  = sum(r["gr_tokens"]  for r in results) / total
    avg_fr_tokens  = sum(r["fr_tokens"]  for r in results) / total

    sep = "=" * 60
    print(f"\n{sep}")
    print("BENCHMARK SUMMARY")
    print(sep)
    print(f"\n{'Metric':<35} {'GraphRAG':>10} {'Flat RAG':>10}")
    print("─" * 57)
    print(f"{'Overall Accuracy':<35} {gr_acc_total:>9.1f}% {fr_acc_total:>9.1f}%")
    print(f"{'Simple Q Accuracy (Q1-10)':<35} {gr_acc_simple:>9.1f}% {fr_acc_simple:>9.1f}%")
    print(f"{'Multi-hop Q Accuracy (Q11-20)':<35} {gr_acc_multi:>9.1f}% {fr_acc_multi:>9.1f}%")
    print(f"{'Avg Latency':<35} {avg_gr_latency:>9.2f}s {avg_fr_latency:>9.2f}s")
    print(f"{'Avg Tokens per Query':<35} {avg_gr_tokens:>9.0f}  {avg_fr_tokens:>9.0f}")
    print(f"\nFlat RAG hallucinated, GraphRAG correct: {len(halluc)}/{total} cases")
    print(f"Multi-hop accuracy gap (GR - FR): {gr_acc_multi - fr_acc_multi:+.1f}%")

    if halluc:
        print("\n--- Failure Modes (Flat RAG sai, GraphRAG dung) ---")
        for r in halluc:
            print(f"\n  Q{r['id']} [{r['type']}]: {r['question']}")
            print(f"  Flat RAG: {r['fr_answer'][:120]}...")
            print(f"  Why wrong: {r['fr_reason']}")
            print(f"  GraphRAG: {r['gr_answer'][:120]}...")

    print(f"\n{sep}")
    print("Goal: GraphRAG multi-hop accuracy > Flat RAG + 20%")
    diff = gr_acc_multi - fr_acc_multi
    if diff >= 20:
        print(f"ACHIEVED: +{diff:.1f}% (target: +20%)")
    else:
        print(f"NOT YET: +{diff:.1f}% (target: +20%)")
    print(sep)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading graph...")
    graph = load_graph()

    print("Loading ChromaDB vector store...")
    col = build_vectorstore()

    print("Loading questions...")
    with open("data/benchmark_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = run_benchmark(questions, graph, col)

    save_csv(results)
    with open("data/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_summary(results)


if __name__ == "__main__":
    main()
