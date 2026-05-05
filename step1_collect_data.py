"""
Buoc 1: Thu thap 100 bai Wikipedia ve cac cong ty / to chuc AI
Luu ket qua vao data/corpus.json
"""

import json
import time
import os
import sys
import wikipediaapi

# Fix encoding cho Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Danh sách 100 công ty / tổ chức AI
AI_COMPANIES = [
    # Big Tech AI Labs
    "OpenAI", "Google DeepMind", "Anthropic", "Meta AI", "Microsoft",
    "Amazon Web Services", "Apple Inc.", "Nvidia", "IBM", "Intel",

    # AI Startups & Research Labs
    "Hugging Face", "Stability AI", "Cohere (company)", "Mistral AI",
    "xAI (company)", "Inflection AI", "Adept (company)", "Character.AI",
    "Runway (company)", "Midjourney",

    # Robotics & Autonomous Systems
    "Boston Dynamics", "Tesla (company)", "Waymo", "Cruise (autonomous vehicle)",
    "Aurora Innovation", "Mobileye", "Zoox", "Nuro (company)",
    "Figure AI", "1X Technologies",

    # Cloud & Infrastructure
    "Google Cloud", "Microsoft Azure", "Amazon (company)", "Oracle Corporation",
    "Salesforce", "ServiceNow", "Palantir Technologies", "Snowflake Inc.",
    "Databricks", "Scale AI",

    # Semiconductor & Hardware
    "Advanced Micro Devices", "Qualcomm", "Arm Holdings",
    "Graphcore", "Cerebras Systems", "SambaNova Systems",
    "Groq (company)", "Tenstorrent",

    # Chinese AI Companies
    "Baidu", "Alibaba Group", "Tencent", "ByteDance",
    "SenseTime", "Megvii", "iFlytek", "Zhipu AI",

    # European AI
    "DeepL", "Aleph Alpha", "BLOOM (language model)",

    # AI in Healthcare
    "Tempus AI", "Recursion Pharmaceuticals", "BenevolentAI",
    "Insilico Medicine", "PathAI",

    # AI Researchers & Foundations
    "Allen Institute for AI", "OpenMined", "EleutherAI",
    "Redwood Research", "Machine Intelligence Research Institute",

    # Enterprise AI
    "UiPath", "Automation Anywhere", "C3.ai",
    "DataRobot", "H2O.ai", "Veritone",

    # Search & NLP
    "Perplexity AI", "You.com", "Neeva",

    # AI Security & Trust
    "Darktrace", "CrowdStrike", "SentinelOne",

    # Semiconductor Ecosystem
    "TSMC", "Samsung Electronics", "SK Hynix",

    # Historical & Foundational
    "DeepMind", "Demis Hassabis", "Geoffrey Hinton",
    "Yann LeCun", "Andrew Ng", "Sam Altman",
    "Yoshua Bengio", "Ian Goodfellow",

    # AI Policy & Standards
    "Partnership on AI", "Center for AI Safety",
    "Future of Life Institute", "AI Now Institute",

    # Voice & Conversational AI
    "Nuance Communications", "SoundHound", "Rasa (company)",

    # Computer Vision
    "Clarifai", "Landing AI", "Roboflow",

    # Additional companies to reach 100
    "Nvidia", "Cerebras Systems", "Graphcore",
    "Stability AI", "Runway (company)", "ElevenLabs",
    "Pika Labs", "Udio", "Suno AI",
    "Luma AI", "Synthesia (company)",
]

# Loại bỏ trùng lặp, giữ thứ tự
seen = set()
UNIQUE_COMPANIES = []
for c in AI_COMPANIES:
    if c not in seen:
        seen.add(c)
        UNIQUE_COMPANIES.append(c)


def collect_corpus(target: int = 100) -> list[dict]:
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="GraphRAG-Lab19/1.0 (VinUniversity student project)"
    )

    corpus = []
    failed = []

    print(f"Bắt đầu thu thập, mục tiêu: {target} bài\n")

    for title in UNIQUE_COMPANIES:
        if len(corpus) >= target:
            break

        page = wiki.page(title)
        if not page.exists():
            print(f"  [SKIP] '{title}' — không tìm thấy")
            failed.append(title)
            continue

        text = page.text
        if len(text) < 200:
            print(f"  [SKIP] '{title}' — bài quá ngắn ({len(text)} ký tự)")
            failed.append(title)
            continue

        # Giới hạn 3000 ký tự đầu để tiết kiệm token khi extract
        corpus.append({
            "id": len(corpus),
            "title": page.title,
            "url": page.fullurl,
            "text": text[:3000],
            "char_count": len(text),
        })
        print(f"  [OK] ({len(corpus):>3}/{target}) {page.title} — {len(text):,} ký tự")

        time.sleep(0.3)  # lịch sự với Wikipedia API

    print(f"\nHoàn thành: {len(corpus)} bài, thất bại: {len(failed)} bài")
    if failed:
        print(f"Danh sách thất bại: {failed}")

    return corpus


def main():
    os.makedirs("data", exist_ok=True)
    output_path = "data/corpus.json"

    corpus = collect_corpus(target=100)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu {len(corpus)} bài vào '{output_path}'")
    print(f"Kích thước file: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
