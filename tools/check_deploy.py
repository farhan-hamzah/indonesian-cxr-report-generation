"""
GPU check of the deploy package (deploy/cxr_report_id, one model at a time): loads it with
plain transformers + peft (no Unsloth), regenerates reports for test images that infer.py
already did, and compares them with infer.py's output, word for word and by Indonesian
metrics against the references. Also measures load time, seconds per image and VRAM.

Unsloth replaces RMSNorm / attention / MLP with its own kernels, so the logits differ
slightly from plain transformers; greedy decoding turns the first differing token into a
different wording for the rest of the report. Exact matches are therefore partial; what has
to hold is the same content and the same metric level on a test set.

    python tools/check_deploy.py                                   # MedGemma, 4-bit (deploy default), 16 images
    python tools/check_deploy.py --model chexone --n 234           # all of test_chexpert: metric parity
    python tools/check_deploy.py --no_4bit --n 8                   # 16-bit variant
"""

import argparse
import difflib
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEPLOY_DIR = PROJECT_DIR / "deploy" / "cxr_report_id"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(DEPLOY_DIR))
from cxr_report import ReportGenerator  # noqa: E402
from rrg.metrics import compute_nlg_metrics  # noqa: E402

TASKS = ("findings", "impression")
RUN = {"medgemma": "medgemma_full", "chexone": "chexone_full"}  # run whose infer.py output is the reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medgemma", choices=list(RUN))
    p.add_argument("--csv", default=None, help="infer.py output to compare with (default: the model's test_chexpert)")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8, help="infer.py's default, so padding matches")
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--save_csv", default=None, help="also write the deploy output (infer.py's columns)")
    args = p.parse_args()

    run = RUN[args.model]
    args.csv = args.csv or str(PROJECT_DIR / f"outputs/{run}/predictions/{run}__test_chexpert.csv")
    ref = pd.read_csv(args.csv).head(args.n)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gen = ReportGenerator(args.model, load_in_4bit=not args.no_4bit)
    load_s = time.time() - t0
    vram_loaded = torch.cuda.memory_allocated() / 2**30
    print(f"[deploy] loaded in {load_s:.0f} s, {vram_loaded:.1f} GiB allocated: {gen.info}")

    # one image alone = the latency of one app request (first call, includes CUDA warm-up)
    view0 = ref.view.iloc[0] if isinstance(ref.view.iloc[0], str) else None
    t0 = time.time()
    gen.generate(ref.image_path.iloc[0], view0)
    first_s = time.time() - t0
    t0 = time.time()
    gen.generate(ref.image_path.iloc[0], view0)
    single_s = time.time() - t0

    rows, t0 = [], time.time()
    for i in range(0, len(ref), args.batch_size):
        b = ref.iloc[i:i + args.batch_size]
        views = [v if isinstance(v, str) else None for v in b.view]
        rows += gen.generate_batch(b.image_path.tolist(), views, return_raw=True)
        print(f"\r[deploy] {min(i + args.batch_size, len(ref))}/{len(ref)} images", end="", flush=True)
    print()
    batch_s = (time.time() - t0) / len(ref)
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30

    print(f"\n{'task':<11} {'exact (raw)':>12} {'exact (post)':>13} {'mean similarity':>16}")
    for task in TASKS:
        raw_eq = [r[f"{task}_raw"] == str(x) for r, x in zip(rows, ref[f"generated_{task}_raw"])]
        post_eq = [r[task] == str(x) for r, x in zip(rows, ref[f"generated_{task}"])]
        sim = [difflib.SequenceMatcher(None, r[task], str(x)).ratio() for r, x in zip(rows, ref[f"generated_{task}"])]
        print(f"{task:<11} {sum(raw_eq):>6}/{len(rows):<5} {sum(post_eq):>7}/{len(rows):<5} {sum(sim) / len(sim):>16.3f}")
        for r, x, ok in zip(rows, ref[f"generated_{task}"], post_eq):
            if not ok:
                print(f"    infer.py: {str(x)[:150]}\n    deploy  : {r[task][:150]}")
                break

    keys = ["bleu_1", "bleu_4", "bleu2_rexrank", "rouge_l", "meteor"]
    print(f"\nIndonesian metrics vs the references on these {len(ref)} images (BERTScore skipped)")
    print(f"{'task':<11} {'run':<9} {'n':>4} " + " ".join(f"{k:>13}" for k in keys))
    for task in TASKS:
        gt = ref[f"ground_truth_{task}"]
        has = (gt.notna() & gt.astype(str).str.strip().ne("")).tolist()
        refs = [str(g) for g, k in zip(gt, has) if k]
        if not refs:
            continue
        runs = {"infer.py": [str(x) if isinstance(x, str) else "" for x, k in zip(ref[f"generated_{task}"], has) if k],
                "deploy": [r[task] for r, k in zip(rows, has) if k]}
        for name, preds in runs.items():
            m = compute_nlg_metrics(preds, refs, bertscore=False)
            print(f"{task:<11} {name:<9} {len(refs):>4} " + " ".join(f"{m[k]:>13.4f}" for k in keys))

    if args.save_csv:
        out = ref.drop(columns=[c for c in ref.columns if c.startswith("generated_")]).copy()
        for task in TASKS:
            out[f"generated_{task}_raw"] = [r[f"{task}_raw"] for r in rows]
            out[f"generated_{task}"] = [r[task] for r in rows]
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.save_csv, index=False)
        print(f"\n[deploy] -> {args.save_csv}")

    print(f"\n[deploy] one image alone (both sections): {first_s:.1f} s first call, {single_s:.1f} s after warm-up")
    print(f"[deploy] batch of {args.batch_size}: {batch_s:.1f} s per image")
    print(f"[deploy] VRAM: {vram_loaded:.1f} GiB after loading, peak {peak:.1f} GiB allocated "
          f"/ {reserved:.1f} GiB reserved (batch {args.batch_size})")


if __name__ == "__main__":
    main()
