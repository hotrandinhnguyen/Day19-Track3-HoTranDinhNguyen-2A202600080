# Lab 19 — GraphRAG vs Flat RAG: Benchmark Report

**Sinh viên:** Hồ Trần Đình Nguyên | **MSSV:** 2A202600080 | **Track:** 3

---

## 1. Tổng quan hệ thống

### 1.1 Corpus
- **100 bài Wikipedia** về các công ty, tổ chức và nhà nghiên cứu AI
- Mỗi bài lấy tối đa **3.000 ký tự** đầu tiên

### 1.2 Knowledge Graph
| Thông số | Giá trị |
|---------|---------|
| Số triples sau dedup | **2.550** |
| Số nodes | **1.802** |
| Số edges | **2.550** |
| Node embeddings | **1.802 nodes × 1536 chiều** (text-embedding-3-small) |
| Predicates chính | FOUNDED_BY, CEO_OF, ACQUIRED_BY, FUNDED_BY, PRODUCT_OF, EMPLOYEE_OF |
| Top nodes by degree | OpenAI (74), Microsoft (54), Waymo (54), Meta AI (50) |

### 1.3 Pipeline GraphRAG
```
Query
  → Entity Extraction (gpt-4o-mini)
  → Seed Node Matching
      Lv1: exact match → partial match → keyword match
      Lv2: cosine similarity với node embeddings (fallback)
  → BFS Subgraph (depth=2)
  → Subgraph-to-Text (seed-adjacent triples ưu tiên trước, max 60)
  → LLM Answer (gpt-4o-mini, chỉ dùng graph context)
```

### 1.4 Pipeline Flat RAG (ChromaDB)
```
Query
  → Vector Search (text-embedding-3-small, top-5 chunks)
  → LLM Answer (gpt-4o-mini, chỉ dùng retrieved chunks)
```

> Cả hai hệ thống đều bị **giới hạn trong cùng corpus 100 bài** — không dùng general knowledge của LLM.

---

## 2. Kết quả Benchmark

### 2.1 Tổng hợp

| Metric | GraphRAG | Flat RAG |
|--------|:--------:|:--------:|
| **Overall Accuracy (20 câu)** | **50%** | 45% |
| **Simple Q Accuracy (Q1–10)** | 40% | 70% |
| **Multi-hop Q Accuracy (Q11–20)** | **60%** | **20%** |
| Avg Latency / query | 3.21s | 2.12s |
| Avg Tokens / query | 1.084 | 551 |
| Flat RAG sai, GraphRAG đúng | **6 cases** | — |
| **Multi-hop accuracy gap** | **GraphRAG +40%** ✅ | — |

> **Mục tiêu đạt được:** GraphRAG vượt Flat RAG **+40%** trên multi-hop (mục tiêu: +20%)

### 2.2 Kết quả từng câu

| ID | Loại | Câu hỏi | GR | FR |
|----|------|---------|:--:|:--:|
| 1 | simple | Who is the CEO of OpenAI? | ✅ | ✅ |
| 2 | simple | What is ChatGPT and who created it? | ❌ | ✅ |
| 3 | simple | What breakthrough did DeepMind achieve with AlphaFold? | ✅ | ❌ |
| 4 | simple | When was Mistral AI founded and where? | ✅ | ✅ |
| 5 | simple | Who owns Waymo? | ✅ | ✅ |
| 6 | simple | What does TSMC do? | ❌ | ✅ |
| 7 | simple | Who co-founded DeepMind? | ❌ | ❌ |
| 8 | simple | What is ByteDance's most famous product? | ❌ | ✅ |
| 9 | simple | Who founded Scale AI? | ❌ | ❌ |
| 10 | simple | What is Hugging Face known for? | ❌ | ✅ |
| 11 | **multi-hop** | John Schulman left OpenAI → joined which company? | ❌ | ❌ |
| 12 | **multi-hop** | Mustafa Suleyman after DeepMind → co-founded what? | ✅ | ❌ |
| 13 | **multi-hop** | Elon Musk left OpenAI → founded what AI company? | ❌ | ✅ |
| 14 | **multi-hop** | Sam Altman leads OpenAI → what products released? | ❌ | ❌ |
| 15 | **multi-hop** | Who funded Perplexity AI? | ✅ | ❌ |
| 16 | **multi-hop** | Meta AI → Llama family models? | ✅ | ❌ |
| 17 | **multi-hop** | Ian Goodfellow left Google DeepMind → joined? | ✅ | ❌ |
| 18 | **multi-hop** | Neeva shut down → acquired by? | ✅ | ✅ |
| 19 | **multi-hop** | Microsoft invested in OpenAI → what else? | ❌ | ❌ |
| 20 | **multi-hop** | Andy Konwinski co-founded which 2 AI companies? | ✅ | ❌ |

---

## 3. Phân tích Failure Modes

### 3.1 GraphRAG thắng Flat RAG — 6 cases

| Q | Path trong graph | Tại sao Flat RAG thua |
|---|-----------------|----------------------|
| Q3 | `Google DeepMind → AlphaFold (PRODUCT_OF)` | Chunk AlphaFold không nằm trong top-5 retrieved |
| Q12 | `Mustafa Suleyman → DeepMind → Inflection AI` | Thông tin ở 2 bài khác nhau, không nối được |
| Q15 | `Perplexity AI → FUNDED_BY → {Jeff Bezos, Nvidia, Databricks, ...}` | Flat RAG chỉ tìm được 3/5 investor, thiếu Nvidia và Databricks |
| Q16 | `Meta AI → PRODUCT_OF → {Llama, Llama 2, Llama 3, Llama 4}` | Flat RAG chỉ trả lời "Llama 1" — không enumerate được cả family |
| Q17 | `Ian Goodfellow → EMPLOYEE_OF → Apple` | Thông tin ở bài Apple, không liên quan đến bài DeepMind |
| Q20 | `Andy Konwinski → {Databricks, Perplexity AI}` | Không có chunk nào mention cả hai cùng lúc |

### 3.2 Flat RAG thắng GraphRAG — simple questions

Flat RAG thắng trên 7/10 câu simple vì:

| Nguyên nhân | Ví dụ |
|-------------|-------|
| Graph thiếu facts cụ thể | TikTok (Douyin), Transformers library — không nằm trong top triples |
| Subgraph context quá rộng | 400+ nodes → 60 triples đầu không cover đủ |
| Judge strict | Cần đúng tất cả key_facts, GraphRAG đúng một phần cũng bị 0 |

### 3.3 Bug tìm được và cách fix

**Bug:** Subgraph OpenAI có 765 edges nhưng chỉ lấy 60 triples đầu **theo thứ tự insert ngẫu nhiên** → triple `Sam Altman CEO_OF OpenAI` bị đẩy ra ngoài top 60.

**Fix:** Ưu tiên triples có seed node là subject hoặc object lên trước, sau đó mới đến triples xa hơn:
```python
# Trước fix: lấy 60 triples đầu theo thứ tự bất kỳ
# Sau fix: direct (seed-adjacent) triples trước, indirect sau
direct = [edges where s or o in seed_set]
indirect = [remaining edges]
context = (direct + indirect)[:60]
```

Kết quả: multi-hop accuracy tăng từ **30% → 60%**.

---

## 4. Chi phí Token và Thời gian

### 4.1 Chi phí xây dựng đồ thị (one-time)

| Bước | Token ước tính | Chi phí | Thời gian |
|------|:-------------:|:-------:|:---------:|
| Extract triples (94 bài) | ~56.400 | ~$0.009 | ~5 phút |
| Node embeddings (1.802 nodes) | ~30.000 | ~$0.003 | ~2 phút |
| ChromaDB index (200 chunks) | ~40.000 | ~$0.004 | ~2 phút |
| **Tổng setup** | **~126.400** | **~$0.016** | **~9 phút** |

### 4.2 Chi phí per query

| Hệ thống | Tokens/query | Chi phí/query | Latency |
|---------|:------------:|:-------------:|:-------:|
| GraphRAG | 1.084 | ~$0.00016 | 3.21s |
| Flat RAG | 551 | ~$0.00008 | 2.12s |

GraphRAG tốn token **gấp ~2x** do graph context (60 triples) trong prompt, nhưng chi phí tuyệt đối vẫn không đáng kể.

---

## 5. Kết luận

### 5.1 Kết quả đạt được

**GraphRAG vượt Flat RAG +40% trên multi-hop questions (60% vs 20%)** — vượt mục tiêu +20% của lab.

### 5.2 Khi nào GraphRAG tỏa sáng

| Loại câu hỏi | Ví dụ | GraphRAG | Flat RAG |
|-------------|-------|:--------:|:--------:|
| Cross-article multi-hop | "Người X rời công ty A → sang công ty B nào?" | ✅ | ❌ |
| Multi-relationship | "X đã được funded bởi những ai?" | ✅ | ❌ |
| Enumerate relationships | "Meta AI đã release những model nào?" | ✅ | ❌ |
| Factual đơn giản một bài | "CEO của X là ai?" (nếu trong top chunks) | ✅/❌ | ✅ |

### 5.3 Key insight

> Điểm mấu chốt không phải graph size hay embedding — mà là **context prioritization**.
> Khi subgraph có 700+ edges nhưng chỉ lấy 60 triples, việc đưa seed-adjacent triples lên đầu
> là yếu tố quyết định giúp multi-hop accuracy tăng từ 30% lên 60%.

### 5.4 Hướng cải thiện tiếp theo

| Cải thiện | Tác động ước tính |
|-----------|:-----------------:|
| Tăng text corpus từ 3k → 8k ký tự/bài | +10–15% simple accuracy |
| Tăng max_triples từ 60 → 100 | +5–10% overall |
| Partial scoring trong judge | Phản ánh đúng hơn |
| Re-ranking triples theo query relevance | +5% multi-hop |
