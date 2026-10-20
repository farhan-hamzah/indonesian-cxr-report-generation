"""Paired comparison of two runs' Indonesian outputs on one test split (CPU only).
usage (from rrg_indo/): python tools/compare_runs.py test_openi [runA runB]   (default medgemma_full chexone_full)
Per-report metrics (bleu2_rexrank, ROUGE-L, METEOR) + corpus BLEU-4, paired bootstrap 95% CI of B - A,
plus output statistics (length, template repetition, English leakage).
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rrg.metrics import bleu2_rexrank  # noqa: E402

split = sys.argv[1]
run_a, run_b = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("medgemma_full", "chexone_full")
A = pd.read_csv(ROOT / f"outputs/{run_a}/predictions/{run_a}__{split}.csv")
B = pd.read_csv(ROOT / f"outputs/{run_b}/predictions/{run_b}__{split}.csv")
assert len(A) == len(B), (len(A), len(B))
B = B.set_index("image_path").loc[A.image_path].reset_index()
assert (A.image_path.values == B.image_path.values).all()

from nltk.translate.meteor_score import meteor_score  # noqa: E402
from rouge_score import rouge_scorer  # noqa: E402
from sacrebleu.metrics import BLEU  # noqa: E402

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
bleu4 = BLEU(max_ngram_order=4, effective_order=True, smooth_method="exp")
EN = re.compile(r"\b(the|of|and|is|are|with|without|there|no|lung|lungs|heart|normal|effusion|seen)\b", re.I)
rng = np.random.default_rng(0)
NB = 5000


def per_sample(refs, preds):
    preds = [p if p.strip() else "." for p in preds]
    return {
        "bleu2_rexrank": np.array([bleu2_rexrank(r, p) for r, p in zip(refs, preds)]),
        "rouge_l": np.array([rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]),
        "meteor": np.array([meteor_score([r.split()], p.split()) for r, p in zip(refs, preds)]),
    }


for task in ("findings", "impression"):
    gt_a, gt_b = A[f"ground_truth_{task}"], B[f"ground_truth_{task}"]
    keep = (gt_a.notna() & gt_a.astype(str).str.strip().ne("")).values
    assert (gt_a[keep].astype(str).values == gt_b[keep].astype(str).values).all(), "references differ"
    refs = gt_a[keep].astype(str).tolist()
    pa = A[f"generated_{task}"][keep].fillna("").astype(str).tolist()
    pb = B[f"generated_{task}"][keep].fillna("").astype(str).tolist()
    n = len(refs)
    print(f"\n=== {split} {task}  n={n}   ({run_a} = A, {run_b} = B)")
    print(f"{'metric':<14} {'A':>7} {'B':>7} {'B-A':>8}  95% CI (paired bootstrap)   P(B>A)")
    ma, mb = per_sample(refs, pa), per_sample(refs, pb)
    idx = rng.integers(0, n, size=(NB, n))
    for k in ma:
        d = mb[k] - ma[k]
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{k:<14} {ma[k].mean():>7.4f} {mb[k].mean():>7.4f} {d.mean():>+8.4f}  [{lo:+.4f}, {hi:+.4f}]   {(boots > 0).mean():.3f}")
    # corpus BLEU-4, paired bootstrap (fewer resamples: corpus scoring is slower)
    ca = bleu4.corpus_score(pa, [refs]).score / 100
    cb = bleu4.corpus_score(pb, [refs]).score / 100
    bd = []
    for row in idx[:(1000 if n < 1000 else 200)]:
        ra = [refs[i] for i in row]
        bd.append(bleu4.corpus_score([pb[i] for i in row], [ra]).score / 100 - bleu4.corpus_score([pa[i] for i in row], [ra]).score / 100)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    print(f"{'bleu_4 corpus':<14} {ca:>7.4f} {cb:>7.4f} {cb - ca:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]   {np.mean(np.array(bd) > 0):.3f}")
    # wins per report on ROUGE-L
    w = (mb["rouge_l"] > ma["rouge_l"]).sum(), (mb["rouge_l"] < ma["rouge_l"]).sum()
    print(f"ROUGE-L per report: B better {w[0]}, A better {w[1]}, tie {n - sum(w)}")
    # output statistics
    for name, preds in ((run_a, pa), (run_b, pb)):
        words = [len(p.split()) for p in preds]
        top, cnt = Counter(preds).most_common(1)[0]
        en = sum(1 for p in preds if EN.search(p))
        print(f"  {name:<14} words mean {np.mean(words):5.1f} (ref {np.mean([len(r.split()) for r in refs]):.1f}) | "
              f"unique {len(set(preds))}/{n} | most common x{cnt}: {top[:70]!r} | English-word hits {en}")
