"""Prove whether a streaming answer blocks the rest of the API.

While a chat answer is streaming (the server holds a global database lock for the whole answer),
fire the endpoints the frontend relies on and measure how long each takes to answer. If they queue
behind the chat, the UI cannot repaint anything until the answer finishes, which is what "frozen"
means.

    .venv/Scripts/python.exe bench_lock_contention.py --port 8040
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(_OPENER)


def _get(path: str, base: str, timeout: float = 120) -> float:
    """Return the time it took for `path` to answer."""
    start = time.perf_counter()
    with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as resp:
        resp.read()
    return (time.perf_counter() - start) * 1000


def _fastest(path: str, base: str, samples: int = 3) -> float:
    """Return the fastest of several readings, after one warm-up.

    A single reading is not a baseline: the server also serves uvicorn's threadpool, and the first
    hit on a cold connection pays setup costs. Taking the fastest of a few measures the work, not
    the noise.
    """
    _get(path, base)
    return min(_get(path, base) for _ in range(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--question", default="WeKnow 的检索内核用了哪些检索路径？")
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    # Baseline: how long do these answer with the server idle? Take the fastest of a few so a stray
    # earlier request cannot inflate the baseline (which would make the ratio below meaningless).
    idle_tasks = _fastest("/api/tasks", base)
    idle_docs = _fastest("/api/documents", base)
    print(f"idle  /api/tasks      {idle_tasks:>8.0f} ms   (fastest of 3)")
    print(f"idle  /api/documents  {idle_docs:>8.0f} ms   (fastest of 3)")
    print("-" * 58)

    # Start a streaming answer in a background thread.
    chat_started = threading.Event()
    chat_done = threading.Event()
    first_token_at: list[float] = []
    probes: dict[str, float] = {}
    repeats: dict[str, float] = {}

    def chat() -> None:
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps({"messages": [{"role": "user", "content": args.question}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            chat_started.set()
            for raw in resp:
                if raw.strip() and not first_token_at:
                    first_token_at.append(time.perf_counter() - t0)
        chat_done.set()

    def probe() -> None:
        chat_started.wait(timeout=30)
        time.sleep(1.0)  # Let the answer get going.
        # These are exactly what the frontend needs while the answer streams.
        probes["tasks"] = _get("/api/tasks", base)
        probes["documents"] = _get("/api/documents", base)

    def probe_repeat() -> None:
        """Sample repeatedly while the answer streams, so one slow reading cannot decide it."""
        chat_started.wait(timeout=30)
        time.sleep(1.0)
        samples: dict[str, list[float]] = {"tasks": [], "documents": []}
        deadline = time.perf_counter() + 25
        while time.perf_counter() < deadline and not chat_done.is_set():
            for key, path in (("tasks", "/api/tasks"), ("documents", "/api/documents")):
                samples[key].append(_get(path, base))
        repeats.update({key: min(values) for key, values in samples.items() if values})

    chatter = threading.Thread(target=chat, daemon=True)
    prober = threading.Thread(target=probe, daemon=True)
    repeater = threading.Thread(target=probe_repeat, daemon=True)
    chatter.start()
    prober.start()
    repeater.start()
    chatter.join(timeout=600)
    prober.join(timeout=600)
    repeater.join(timeout=600)

    print(f"first token           {first_token_at[0] * 1000 if first_token_at else -1:>8.0f} ms")
    print("-" * 58)
    for key in ("tasks", "documents"):
        if key in probes:
            print(f"during chat /api/{key:<12} {probes[key]:>8.0f} ms"
                  f"   ({probes[key] / max(idle_tasks if key == 'tasks' else idle_docs, 1e-9):.0f}x idle)")
    for key in ("tasks", "documents"):
        if key in repeats:
            base_idle = idle_tasks if key == "tasks" else idle_docs
            print(f"best  /api/{key:<12} {repeats[key]:>8.0f} ms"
                  f"   ({repeats[key] / max(base_idle, 1e-9):.1f}x idle)")
    print("-" * 58)
    # Judge on the *best* reading taken during the answer: if even the luckiest request is slow,
    # something is genuinely blocking; a single unlucky timing is not evidence.
    judged = {**probes, **repeats}
    blocked = any(v > 2000 for v in judged.values())  # noqa: PLR2004
    print("VERDICT:", "BLOCKED - the UI cannot repaint while answering" if blocked
          else "not blocked - these endpoints answer promptly")


if __name__ == "__main__":
    main()
