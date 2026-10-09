"""End-to-end chain test against a live `weknow serve` instance.

Drives the real HTTP API and times each stage separately:
  upload -> parse -> index -> retrieve -> answer

Run the server first:
    .venv/Scripts/python.exe -m weknow serve --port 8011
Then:
    .venv/Scripts/python.exe bench_chain.py --port 8011 --file 多模态扩展-OCR+VLM.md
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

# The shell may export HTTP_PROXY/HTTPS_PROXY (this machine does), which makes urllib route even
# 127.0.0.1 through the proxy and return 502. Loopback must never be proxied.
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(_NO_PROXY)

PENDING = {"uploading", "parsing", "embedding"}


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())


def _upload(base: str, path: Path) -> str:
    boundary = "----weknowchain"
    data = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{base}/api/documents",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["task_id"]


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--file", default="多模态扩展-OCR+VLM.md")
    parser.add_argument("--question", default="这个扩展支持哪些 OCR 引擎？")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    path = Path(args.file)
    print(f"target  = {base}")
    print(f"file    = {path.name} ({path.stat().st_size} bytes)")
    print("-" * 62)

    # ---- 1. upload -------------------------------------------------------
    t0 = time.perf_counter()
    task_id = _upload(base, path)
    upload_ms = (time.perf_counter() - t0) * 1000
    print(f"{'1. upload (POST accepted)':<34} {upload_ms:>8.0f} ms  task={task_id}")

    # ---- 2. parse + index (poll the in-memory task endpoint) -------------
    t1 = time.perf_counter()
    seen: list[tuple[float, str]] = []
    final = None
    while True:
        tasks = _get_json(f"{base}/api/tasks")["tasks"]
        task = next((t for t in tasks if t["task_id"] == task_id), None)
        if task is None:
            print("task vanished")
            break
        if not seen or seen[-1][1] != task["status"]:
            seen.append((time.perf_counter() - t1, task["status"]))
        if task["status"] not in PENDING:
            final = task
            break
        time.sleep(0.25)
    index_ms = (time.perf_counter() - t1) * 1000
    transitions = " -> ".join(f"{s}@{ms:.0f}ms" for ms, s in seen)
    print(f"{'2. parse + index (to ready)':<34} {index_ms:>8.0f} ms  {transitions}")
    print(f"   final status = {final['status']}  detail = {final['detail']!r}")
    if final["status"] != "ready":
        raise SystemExit(f"indexing did not succeed: {final}")

    # ---- 3. confirm the document is queryable ----------------------------
    t2 = time.perf_counter()
    docs = _get_json(f"{base}/api/documents")["documents"]
    doc = next((d for d in docs if d["filename"] == path.name), None)
    print(f"{'3. list documents':<34} {(time.perf_counter() - t2) * 1000:>8.0f} ms  {len(docs)} docs")
    print(f"   stored: {doc}")
    # ---- 4. retrieval ----------------------------------------------------
    t3 = time.perf_counter()
    search = _post_json(f"{base}/api/search", {"query": args.question, "top_k": 5})
    search_ms = (time.perf_counter() - t3) * 1000
    hits = search.get("results", search.get("chunks", []))
    print(f"{'4. retrieval (hybrid + rerank)':<34} {search_ms:>8.0f} ms  {len(hits)} hits")
    for i, hit in enumerate(hits[:3], 1):
        score = hit.get("score", hit.get("rerank_score", "?"))
        body = (hit.get("body") or hit.get("content") or "")[:70].replace("\n", " ")
        print(f"   #{i} score={score}  {body}...")

    # ---- 5. generation ---------------------------------------------------
    t4 = time.perf_counter()
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=json.dumps({"messages": [{"role": "user", "content": args.question}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    tokens, sources, first_token_ms = 0, 0, None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "token" in event:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t4) * 1000
                tokens += 1
            if "sources" in event:
                sources = len(event["sources"])
    gen_ms = (time.perf_counter() - t4) * 1000
    ttft = f"{first_token_ms:.0f} ms" if first_token_ms else "n/a"
    print(f"{'5. answer (SSE to [DONE])':<34} {gen_ms:>8.0f} ms  TTFT={ttft}")
    print(f"   tokens streamed = {tokens}, sources cited = {sources}")

    print("-" * 62)
    print(f"total chain = {upload_ms + index_ms + search_ms + gen_ms:.0f} ms "
          f"(upload {upload_ms:.0f} + index {index_ms:.0f} + retrieve {search_ms:.0f} "
          f"+ generate {gen_ms:.0f})")
    print("machine-readable:")
    print(json.dumps({
        "upload_ms": round(upload_ms),
        "index_ms": round(index_ms),
        "search_ms": round(search_ms),
        "generate_ms": round(gen_ms),
        "ttft_ms": round(first_token_ms) if first_token_ms else None,
        "chunks": doc["chunks"] if doc else None,
        "hits": len(hits),
        "tokens": tokens,
        "sources": sources,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
