# Lab 19 — GraphRAG với Tech Company Corpus

**Sinh viên:** Hồ Trần Đình Nguyên | **MSSV:** 2A202600080 | **Track:** 3

---

## Mục tiêu

Xây dựng hệ thống **GraphRAG** trên corpus 100 bài Wikipedia về các công ty AI, so sánh với **Flat RAG** (ChromaDB), và chứng minh GraphRAG vượt trội trên các câu hỏi multi-hop.

## Cấu trúc project

```
├── step1_collect_data.py    # Thu thập 100 bài Wikipedia
├── step2b_reextract.py      # Trích xuất triples bằng gpt-4o-mini
├── step3_build_graph.py     # Xây đồ thị NetworkX + node embeddings
├── step4_graphrag.py        # Pipeline GraphRAG
├── step4b_flat_rag.py       # Pipeline Flat RAG (ChromaDB)
├── step5_benchmark.py       # Benchmark 20 câu hỏi
├── report.md                # Báo cáo kết quả & phân tích
└── data/
    ├── corpus.json          # 100 bài Wikipedia
    ├── triples_v2.json      # 2.727 triples
    ├── graph.gpickle        # Knowledge graph (NetworkX)
    ├── graph_openai.png     # Visualization subgraph OpenAI
    ├── graph_deepmind.png   # Visualization subgraph DeepMind
    ├── graph_overview.png   # Toàn cảnh đồ thị
    └── benchmark_results.csv
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install networkx matplotlib openai python-dotenv
pip install wikipedia-api chromadb
```

Tạo file `.env`:
```
OPENAI_API_KEY=sk-...
```

## Chạy từng bước

```bash
# Bước 1: Thu thập dữ liệu
python step1_collect_data.py

# Bước 2: Trích xuất triples
python step2b_reextract.py

# Bước 3: Xây đồ thị + node embeddings
python step3_build_graph.py

# Bước 4: Test pipeline (optional)
python step4_graphrag.py

# Bước 5: Chạy benchmark
python step5_benchmark.py
```

## Kiến trúc GraphRAG

```
Query
  → Entity Extraction (gpt-4o-mini)
  → Seed Node Matching (fuzzy + cosine similarity với node embeddings)
  → BFS Traversal (depth=2, max 60 triples)
  → Subgraph-to-Text
  → LLM Answer (gpt-4o-mini)
```

## Kết quả

| Metric | GraphRAG | Flat RAG |
|--------|:--------:|:--------:|
| Multi-hop Accuracy | **60%** | **20%** |
| Overall Accuracy | 50% | 45% |
| Avg Latency | 3.21s | 2.12s |
| Avg Tokens/query | 1.084 | 551 |

GraphRAG vượt Flat RAG **+40%** trên multi-hop questions ✅ (mục tiêu: +20%)

## Chi phí ước tính

| Hạng mục | Chi phí |
|---------|---------|
| Xây đồ thị (one-time) | ~$0.015 |
| Benchmark 20 câu (×5 API calls/câu) | ~$0.01 |
| **Tổng** | **~$0.025** |
