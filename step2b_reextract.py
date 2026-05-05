"""
Re-extract triples: 50/bai, prompt chi tiet hon, luu vao data/triples_v2.json
"""

import json, os, sys, time
from openai import OpenAI
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a knowledge graph extractor for AI companies and researchers.
Extract factual triples from the article as (subject, predicate, object).

Predicates to use:
- FOUNDED_BY, CO_FOUNDED_BY  → who started the company
- FOUNDED_IN                 → founding year
- CEO_OF, CTO_OF, CHAIRMAN_OF → leadership roles
- ACQUIRED_BY, MERGED_WITH   → M&A events
- INVESTED_IN, FUNDED_BY     → investment relationships
- PRODUCT_OF, CREATED_BY     → products and creators
- LOCATED_IN                 → headquarters city/country
- PARTNER_OF, COLLABORATES_WITH → partnerships
- COMPETITOR_OF              → competitors
- EMPLOYEE_OF, WORKS_AT      → employment (past or present)
- SUBSIDIARY_OF, PART_OF     → ownership structure
- TRAINED_ON                 → datasets used
- BASED_ON                   → model architecture lineage
- WON_AWARD                  → prizes and recognition
- SANCTIONED_BY              → government actions

Rules:
- Extract UP TO 50 triples
- subject and object must be specific named entities, NOT generic terms
- Include founding year, CEO name, key products, investors, location
- Output ONLY valid JSON array, no explanation

Example output:
[
  {"s": "OpenAI", "p": "FOUNDED_BY", "o": "Sam Altman"},
  {"s": "OpenAI", "p": "FOUNDED_IN", "o": "2015"},
  {"s": "OpenAI", "p": "CEO_OF", "o": "Sam Altman"},
  {"s": "ChatGPT", "p": "PRODUCT_OF", "o": "OpenAI"}
]"""


def extract_triples(text: str, title: str) -> list[dict]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Article title: {title}\n\n{text[:2500]}"},
            ],
            temperature=0,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        triples = json.loads(raw)
        return [t for t in triples if all(k in t for k in ("s", "p", "o"))]
    except Exception as e:
        print(f"    [ERROR] {title}: {e}")
        return []


def main():
    output_path = "data/triples_v2.json"
    corpus_path = "data/corpus.json"

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Resume support
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_triples = json.load(f)
        done_ids = {t["source_id"] for t in all_triples}
        print(f"Resume: {len(done_ids)} bai, {len(all_triples)} triples")
    else:
        all_triples = []
        done_ids = set()

    total = len(corpus)
    for article in corpus:
        aid = article["id"]
        if aid in done_ids:
            continue

        triples = extract_triples(article["text"], article["title"])
        for t in triples:
            t["source_id"] = aid
            t["source_title"] = article["title"]
        all_triples.extend(triples)
        done_ids.add(aid)

        done_count = len(done_ids)
        print(f"  [{done_count:>3}/{total}] {article['title']} — {len(triples)} triples")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_triples, f, ensure_ascii=False, indent=2)

        time.sleep(0.5)

    print(f"\nHoan thanh: {len(all_triples)} triples tu {total} bai")

    from collections import Counter
    pred_counts = Counter(t["p"] for t in all_triples)
    print("\nTop predicates:")
    for pred, count in pred_counts.most_common(15):
        print(f"  {pred:30s} {count:>4}")


if __name__ == "__main__":
    main()
