"""Paired ReXrank comparison of two runs at study level (CPU only, reads the per-sample parquets).
usage (from rrg_indo/rexrank_eval): python compare_runs.py [runA runB] [--json_b out.json]
Prints means per split/track and a paired bootstrap 95% CI of B - A over the same studies.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from rexrank import summarize  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
run_a, run_b = (args[0], args[1]) if len(args) >= 2 else ("medgemma_full", "chexone_full")
A = pd.read_parquet(f"../outputs/{run_a}/rexrank/{run_a}__rexrank_per_sample.parquet")
B = pd.read_parquet(f"../outputs/{run_b}/rexrank/{run_b}__rexrank_per_sample.parquet")
if "--json_b" in sys.argv:
    llm = [m for m in ("green", "crimson") if m in B]
    json.dump(summarize(B, llm), open(sys.argv[sys.argv.index("--json_b") + 1], "w"), indent=1)

key = ["split", "track", "image_path"]
m = A[A.study_rep].merge(B[B.study_rep], on=key, suffixes=("_a", "_b"))
assert (m.ref_a == m.ref_b).all(), "references differ"
metrics = [k for k in ("radgraph", "ratescore", "semb_score", "bleu2", "bertscore", "bertscore_cxrm", "radcliq_v1",
                       "radcliq_v1_cxrm") if f"{k}_a" in m and f"{k}_b" in m]
inverse = {k: c for k, c in (("1/radcliq", "radcliq_v1"), ("1/radcliq_cxrm", "radcliq_v1_cxrm")) if c in metrics}
llm = [x for x in ("green", "crimson") if f"{x}_a" in m and f"{x}_b" in m]
rng = np.random.default_rng(0)
rng_cxrm = np.random.default_rng(1)  # own draws, so adding the _cxrm variant left the other metrics' CIs unchanged
NB = 2000
for split in ("test_chexpert", "test_mimic", "test_openi"):
    for track in ("findings", "impression", "findings_impression"):
        g = m[(m.split == split) & (m.track == track)]
        n = len(g)
        if not n:
            continue
        print(f"\n=== {split} {track}  studies={n}  (A={run_a}, B={run_b})")
        print(f"{'metric':<15} {'A':>7} {'B':>7} {'B-A':>8}  95% CI            better")
        idx = rng.integers(0, n, size=(NB, n))
        for k in metrics + list(inverse) + llm:
            if k in inverse:
                a, b = g[f"{inverse[k]}_a"].values, g[f"{inverse[k]}_b"].values
                va, vb = 1 / a.mean(), 1 / b.mean()
                boots = 1 / b[idx].mean(1) - 1 / a[idx].mean(1)
            else:
                sub = g[[f"{k}_a", f"{k}_b"]].dropna()
                a, b = sub[f"{k}_a"].values, sub[f"{k}_b"].values
                if not len(a):
                    continue
                va, vb = a.mean(), b.mean()
                ii = (rng_cxrm if k.endswith("_cxrm") else rng).integers(0, len(a), size=(NB, len(a)))
                boots = b[ii].mean(1) - a[ii].mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            lower_better = k.startswith("radcliq_v1")
            sig = lo > 0 or hi < 0
            if sig:
                b_wins = (vb < va) if lower_better else (vb > va)
                win = run_b if b_wins else run_a
            else:
                win = "tie (CI spans 0)"
            print(f"{k:<15} {va:>7.4f} {vb:>7.4f} {vb - va:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {win}")
