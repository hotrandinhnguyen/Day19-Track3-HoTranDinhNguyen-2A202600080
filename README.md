# Lab 19 — Xây dựng hệ thống GraphRAG với Tech Company Corpus

**Sinh viên:** Hồ Trần Đình Nguyên | **MSSV:** 2A202600080 | **Track:** 3 | **VinUniversity**

---

## Mục tiêu

Xây dựng pipeline **GraphRAG** hoàn chỉnh trên corpus 100 bài Wikipedia về các công ty AI:
1. Trích xuất thực thể và quan hệ (Entity & Relation Extraction)
2. Xây dựng đồ thị tri thức với NetworkX + node embeddings
3. Implement GraphRAG retrieval: BFS traversal → LLM answer
4. So sánh với Flat RAG (ChromaDB) trên 20 câu hỏi benchmark

**Kết quả:** GraphRAG vượt Flat RAG **+40%** trên multi-hop questions (60% vs 20%) ✅

---

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install networkx matplotlib openai python-dotenv wikipedia-api chromadb
```

Tạo file `.env`:
```
OPENAI_API_KEY=sk-...
```

---

## Chạy từng bước

```bash
# Bước 1: Thu thập 100 bài Wikipedia về công ty AI
python step1_collect_data.py

# Bước 2: Trích xuất triples (subject, predicate, object) bằng gpt-4o-mini
python step2b_reextract.py

# Bước 3: Xây đồ thị NetworkX + thêm embeddings cho nodes
python step3_build_graph.py

# Bước 4 (optional): Test pipeline
python step4_graphrag.py

# Bước 5: Chạy benchmark 20 câu hỏi
python step5_benchmark.py
```

---

## Kiến trúc

### GraphRAG Pipeline
```
Query
  → Entity Extraction      (gpt-4o-mini)
  → Seed Node Matching     (fuzzy match + cosine similarity với node embeddings)
  → BFS Subgraph           (depth=2, ưu tiên seed-adjacent triples)
  → Subgraph-to-Text       (max 60 triples)
  → LLM Answer             (gpt-4o-mini, chỉ dùng graph context)
```

### Flat RAG Pipeline
```
Query
  → Vector Search          (ChromaDB, text-embedding-3-small, top-5 chunks)
  → LLM Answer             (gpt-4o-mini, chỉ dùng retrieved chunks)
```

---

## Knowledge Graph

| Thông số | Giá trị |
|---------|---------|
| Corpus | 100 bài Wikipedia |
| Triples | 2.550 (sau deduplication) |
| Nodes | 1.802 |
| Node embeddings | 1.802 × 1536 chiều |
| Predicates | FOUNDED_BY, CEO_OF, ACQUIRED_BY, FUNDED_BY, PRODUCT_OF, EMPLOYEE_OF, ... |

---

## Kết quả Benchmark

| Metric | GraphRAG | Flat RAG |
|--------|:--------:|:--------:|
| **Multi-hop Accuracy** | **60%** | 20% |
| Simple Accuracy | 40% | 70% |
| Overall Accuracy | **50%** | 45% |
| Avg Latency | 3.21s | 2.12s |
| Avg Tokens/query | 1.084 | 551 |
| **Multi-hop gap** | **+40%** ✅ | — |

---

## Deliverables

| File | Mô tả |
|------|-------|
| `step1_collect_data.py` | Thu thập corpus |
| `step2b_reextract.py` | Trích xuất triples |
| `step3_build_graph.py` | Xây đồ thị + embeddings |
| `step4_graphrag.py` | GraphRAG pipeline |
| `step4b_flat_rag.py` | Flat RAG ChromaDB |
| `step5_benchmark.py` | Benchmark script |
| `data/graph_openai.png` | Đồ thị subgraph OpenAI |
| `data/graph_overview.png` | Toàn cảnh đồ thị |
| `data/benchmark_results.csv` | Kết quả 20 câu hỏi |
| `report.md` | Báo cáo phân tích |
