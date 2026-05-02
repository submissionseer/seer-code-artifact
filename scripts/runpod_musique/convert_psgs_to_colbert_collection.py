#!/usr/bin/env python3
"""
Convert DPR wiki20M passages (psgs_w100.tsv.gz) into a ColBERT collection.tsv
that is compatible with FrugalRAG's title parsing logic: "title | text".

Input:  psgs_w100.tsv.gz  (columns typically: id, text, title)
Output: collection.tsv    (pid<TAB>title | text)
"""

from __future__ import annotations

import csv
import gzip
import os
import sys
import time


TOTAL_ESTIMATE = 21_015_324  # DPR wikipedia_split/psgs_w100 count (approx)


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: convert_psgs_to_colbert_collection.py <psgs_w100.tsv.gz> <collection.tsv>",
            file=sys.stderr,
        )
        return 2

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start = time.time()
    count = 0

    with gzip.open(in_path, "rt", encoding="utf-8", newline="") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        for line_idx, row in enumerate(reader):
            # ColBERT's collection loader expects the first TSV column to be a
            # contiguous 0-based integer matching the line index.
            pid = str(line_idx)
            title = clean(row.get("title"))
            text = clean(row.get("text"))
            combined = f"{title} | {text}" if title else text
            fout.write(f"{pid}\t{combined}\n")
            count += 1

            if count % 100_000 == 0:
                elapsed = time.time() - start
                rate = count / max(elapsed, 1e-9)
                remaining = max(TOTAL_ESTIMATE - count, 0)
                eta_sec = remaining / max(rate, 1e-9)
                print(
                    f"[convert] rows={count:,} rate={rate:,.0f}/s "
                    f"elapsed={elapsed/3600:.2f}h eta={eta_sec/3600:.2f}h",
                    flush=True,
                )

    elapsed = time.time() - start
    print(f"[convert] DONE rows={count:,} elapsed={elapsed/3600:.2f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
