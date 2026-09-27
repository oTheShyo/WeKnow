#!/usr/bin/env python
"""端到端生成链路基准 + 小评测集问答准确率（RAGLite）。

前置：需要一个 OpenAI 兼容的 LLM 端点（生成）和一个 embedding 端点（建库/查询）。
你的 CC Switch 本地代理即可：设置下面三个环境变量后直接跑。

环境变量：
  OPENAI_API_BASE   例如 http://127.0.0.1:<ccswitch_port>/v1
  OPENAI_API_KEY    你的 USTC / CC Switch key
  RAG_LLM_MODEL     生成模型名，例如 qwen-plus / deepseek-chat / glm-4
  RAG_EMBED_MODEL   嵌入模型名，例如 text-embedding-v3（需端点支持 embedding）
  RAG_RERANKER      可选：留空则用 reranker=None（跳过 Cross-Encoder，纯测生成链路）

运行（先 dry-run 检查环境，再正式跑）：
  ./.venv/Scripts/python.exe gen_bench.py --check-env
  ./.venv/Scripts/python.exe -u gen_bench.py --runs 10 --db gen_bench.db 2>&1 | tee gen_bench_log.txt

说明：
  - 端到端 P50/P95 同时统计「首字延迟 TTFT」与「全文完成延迟」。
  - 问答准确率用一个内置小评测集（答案唯一 span 子串匹配），是可辩护的 sanity-check，
    不是完整 RAGAS 集；如需更严谨可把 QA 集换成你自己的标注集。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

# ---------------------------------------------------------------------------
# 内置小知识库 + 评测集（答案均为文档中唯一可定位的 span，便于子串匹配判定）
# ---------------------------------------------------------------------------
SAMPLE_DOC = """# 量子计算基础

量子比特（qubit）是量子计算的基本信息单元，与经典比特只能取 0 或 1 不同，qubit 可以处于 0 和 1 的叠加态。
超导量子计算机使用约瑟夫森结在接近绝对零度的环境下实现 qubit。
Shor 算法能够在多项式时间内分解大整数，对现有公钥密码体系构成潜在威胁。
Grover 算法在无序数据库中搜索目标，相较经典算法获得平方根级的加速。
量子纠错通过冗余编码保护脆弱的量子态免受退相干影响，是实用化量子计算的关键难题。
"""

QA_SET = [
    ("量子计算的基本信息单元是什么？", "量子比特"),
    ("qubit 和经典比特的根本区别是什么？", "叠加态"),
    ("超导量子计算机用什么实现 qubit？", "约瑟夫森结"),
    ("哪个算法能在多项式时间内分解大整数？", "Shor"),
    ("Grover 算法相比经典搜索获得什么加速？", "平方根"),
    ("实用化量子计算的关键难题是什么？", "量子纠错"),
]


def build_config() -> object:
    from raglite import RAGLiteConfig

    base = os.environ.get("OPENAI_API_BASE", "")
    key = os.environ.get("OPENAI_API_KEY", "")
    llm = os.environ.get("RAG_LLM_MODEL", "")
    embed = os.environ.get("RAG_EMBED_MODEL", "")
    reranker = os.environ.get("RAG_RERANKER", "") or None

    if not (base and key and llm and embed):
        raise SystemExit(
            "缺少环境变量：需设置 OPENAI_API_BASE / OPENAI_API_KEY / "
            "RAG_LLM_MODEL / RAG_EMBED_MODEL（reranker 可选）。先跑 --check-env。"
        )
    # litellm 用 "openai/<model>" + OPENAI_API_BASE/KEY 走兼容端点
    return RAGLiteConfig(
        db_url=os.environ.get("RAG_DB_URL", "duckdb:///gen_bench.db"),
        llm=f"openai/{llm}",
        embedder=f"openai/{embed}",
        reranker=reranker,
    )


def insert_sample(config: object) -> None:
    from raglite import Document, insert_documents

    doc = Document.from_text(SAMPLE_DOC, author="bench", topic="quantum", year=2026)
    insert_documents([doc], config=config)
    print("[insert] sample doc inserted.")


def bench_generation(config: object, runs: int) -> dict:
    from raglite import rag

    ttfts, totals = [], []
    for i in range(runs):
        messages = [{"role": "user", "content": "请用一句话解释什么是量子比特。"}]
        t0 = time.perf_counter()
        first_ts = None
        n_tokens = 0
        for tok in rag(messages, config=config):
            n_tokens += 1
            if first_ts is None:
                first_ts = time.perf_counter()
        total = time.perf_counter() - t0
        ttft = (first_ts - t0) * 1e3 if first_ts else float("nan")
        ttfts.append(ttft)
        totals.append(total * 1e3)
        print(f"[gen] run {i}: tokens={n_tokens} TTFT={ttft:.0f}ms total={totals[-1]:.0f}ms")
    return {
        "runs": runs,
        "ttft_p50_ms": round(statistics.median(ttfts), 1),
        "ttft_p95_ms": round(sorted(ttfts)[max(0, int(0.95 * (runs - 1)))], 1),
        "total_p50_ms": round(statistics.median(totals), 1),
        "total_p95_ms": round(sorted(totals)[max(0, int(0.95 * (runs - 1)))], 1),
    }


def bench_qa_accuracy(config: object) -> dict:
    from raglite import rag

    correct = 0
    details = []
    for q, gold in QA_SET:
        messages = [{"role": "user", "content": q}]
        answer = "".join(rag(messages, config=config))
        ok = gold.lower() in answer.lower()
        correct += int(ok)
        details.append({"q": q, "gold": gold, "hit": ok})
        print(f"[qa] {'OK ' if ok else 'MISS'} | {q} -> expects '{gold}'")
    return {"n": len(QA_SET), "correct": correct, "accuracy": round(correct / len(QA_SET), 3), "details": details}


def check_env() -> None:
    need = ["OPENAI_API_BASE", "OPENAI_API_KEY", "RAG_LLM_MODEL", "RAG_EMBED_MODEL"]
    print("== env check ==")
    for k in need:
        v = os.environ.get(k)
        print(f"  {k:18s}: {'<set>' if v else '<MISSING>'}")
    print(f"  RAG_RERANKER     : {os.environ.get('RAG_RERANKER', '(none -> reranker=None)')}")
    print("若四项均 <set>，即可正式运行；否则先 export 对应变量（CC Switch 端点）。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-env", action="store_true", help="只检查环境变量并打印配置")
    ap.add_argument("--runs", type=int, default=10, help="端到端生成重复次数")
    ap.add_argument("--db", default="gen_bench.db", help="基准用数据库文件")
    ap.add_argument("--skip-insert", action="store_true", help="假定文档已入库，跳过插入")
    args = ap.parse_args()

    if args.check_env:
        check_env()
        return

    os.environ.setdefault("RAG_DB_URL", f"duckdb:///{args.db}")
    config = build_config()
    if not args.skip_insert:
        insert_sample(config)
    print("== generation benchmark ==")
    gen = bench_generation(config, args.runs)
    print("== qa accuracy ==")
    qa = bench_qa_accuracy(config)
    result = {"generation": gen, "qa_accuracy": qa}
    print("\n=== RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
