"""
Indonesian NLG metrics (rrg/metrics.py, 0-1 scale) for infer.py output, per test split and per
track: findings, impression, and findings+impression (both references present).

Scores generated_<task> (post-processed by infer.py); --raw scores the untouched
generated_<task>_raw columns instead, to see what post-processing changed.
--no_bertscore skips BERTScore but keeps bertscore_f1 from an existing metrics JSON of the
same predictions (fast re-scoring after a metric change).

    python evaluate.py --predictions_csv outputs/medgemma_full/predictions/medgemma_full__test_mimic.csv \
        --metric_dir outputs/medgemma_full/metrics
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from rrg.config import METRIC_DIR
from rrg.metrics import compute_nlg_metrics


def track(df: pd.DataFrame, name: str, bertscore: bool) -> dict | None:
    def has(col):
        return df[col].notna() & df[col].astype(str).str.strip().ne("")

    if name == "findings_impression":
        need = ["ground_truth_findings", "ground_truth_impression", "generated_findings", "generated_impression"]
        if not all(c in df for c in need):
            return None
        sub = df[has("ground_truth_findings") & has("ground_truth_impression")]
        preds = (sub["generated_findings"].fillna("") + " " + sub["generated_impression"].fillna("")).tolist()
        refs = (sub["ground_truth_findings"] + " " + sub["ground_truth_impression"]).tolist()
    else:
        if f"generated_{name}" not in df:
            return None
        sub = df[has(f"ground_truth_{name}")]
        preds = sub[f"generated_{name}"].fillna("").astype(str).tolist()
        refs = sub[f"ground_truth_{name}"].astype(str).tolist()
    if sub.empty:
        return None
    return {"n": len(sub), **compute_nlg_metrics(preds, refs, bertscore)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--no_bertscore", action="store_true")
    p.add_argument("--raw", action="store_true", help="score generated_<task>_raw (before post-processing)")
    p.add_argument("--metric_dir", default=None, help=f"default {METRIC_DIR}")
    args = p.parse_args()

    df = pd.read_csv(args.predictions_csv)
    if args.raw:
        for task in ("findings", "impression"):
            if f"generated_{task}_raw" in df:
                df[f"generated_{task}"] = df[f"generated_{task}_raw"]
    results = {}
    for split, sdf in df.groupby("split"):
        results[split] = {}
        for name in ("findings", "impression", "findings_impression"):
            m = track(sdf, name, not args.no_bertscore)
            if m:
                results[split][name] = m
                print(f"[{split}] {name:20s} n={m['n']:5d}  BLEU-1={m['bleu_1']:.4f}  BLEU-4={m['bleu_4']:.4f}  "
                      f"BLEU-2(ReXrank)={m['bleu2_rexrank']:.4f}  "
                      f"ROUGE-L={m['rouge_l']:.4f}  METEOR={m['meteor']:.4f}"
                      + (f"  BERTScore={m['bertscore_f1']:.4f}" if "bertscore_f1" in m else ""))

    metric_dir = Path(args.metric_dir) if args.metric_dir else METRIC_DIR
    metric_dir.mkdir(parents=True, exist_ok=True)
    out_path = metric_dir / f"{Path(args.predictions_csv).stem}{'_raw' if args.raw else ''}_metrics.json"
    if args.no_bertscore and out_path.exists():  # keep BERTScore from an earlier run of the same predictions
        old = json.loads(out_path.read_text())
        for split, tracks in results.items():
            for name, m in tracks.items():
                prev = old.get(split, {}).get(name, {})
                if "bertscore_f1" in prev and prev.get("n") == m["n"]:
                    m["bertscore_f1"] = prev["bertscore_f1"]
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
