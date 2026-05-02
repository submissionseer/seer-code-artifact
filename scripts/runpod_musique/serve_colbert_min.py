#!/usr/bin/env python3
import argparse
from flask import Flask, request, jsonify
from colbert import Searcher

app = Flask(__name__)
_searcher = None
_collection = None


def parse_title_and_text(raw: str):
    if raw is None:
        return "", ""
    if " | " in raw:
        title, text = raw.split(" | ", 1)
        return title.strip(), text.strip()
    return "", str(raw)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/search")
def api_search():
    q = request.args.get("query", "")
    if not q:
        return jsonify({"error": "missing query"}), 400
    k = int(request.args.get("k", "10"))
    pids, ranks, scores = _searcher.search(q, k=k)

    topk = []
    for pid, rank, score in zip(pids, ranks, scores):
        raw = _collection[pid]
        title, text = parse_title_and_text(raw)
        topk.append(
            {
                "rank": int(rank),
                "pid": int(pid),
                "score": float(score),
                "title": title,
                "text": text,
            }
        )

    return jsonify({"query": q, "k": k, "topk": topk})


def main():
    global _searcher, _collection

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_root", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threaded", action="store_true")
    args = parser.parse_args()

    print(
        f"Loading Searcher(index={args.index}, index_root={args.index_root}, checkpoint={args.checkpoint})",
        flush=True,
    )
    _searcher = Searcher(
        index=args.index,
        index_root=args.index_root,
        checkpoint=args.checkpoint,
        collection=args.collection,
    )
    _collection = _searcher.collection
    print("Searcher ready", flush=True)

    app.run(host=args.host, port=args.port, threaded=args.threaded)


if __name__ == "__main__":
    main()
