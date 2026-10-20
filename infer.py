"""
Batch report generation with a trained adapter (any model in rrg/models.py; the base model
is read from the adapter's run_args.json / adapter_config.json). Writes one CSV row per image
with ground truth and generated findings/impression; partial results are flushed every
--save_every batches and --resume skips images already done.
generated_<task> is post-processed (rrg/postprocess.py: repeated sentences removed),
generated_<task>_raw is the untouched model output.

    python infer.py --adapter checkpoints/medgemma_full/final_adapter --split test_chexpert
    python infer.py --adapter checkpoints/chexone_full/final_adapter --split test_mimic --resume
    python infer.py --adapter StanfordAIMI/CheXOne --split test_openi --limit 16     # zero-shot base model
    python infer.py --adapter ... --image_path some/cxr.png --view PA
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from rrg.config import MANIFEST_PATH, PRED_DIR, TASKS
from rrg.data import load_image
from rrg.models import spec_for_adapter
from rrg.postprocess import postprocess
from rrg.prompts import build_prompt, load_hints


def load_model(adapter: str, load_in_4bit: bool = True):
    import unsloth  # noqa: F401 - must be imported before transformers
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(adapter, load_in_4bit=load_in_4bit)
    FastVisionModel.for_inference(model)
    processor.tokenizer.padding_side = "left"  # decoder-only batch generation
    return model, processor


@torch.no_grad()
def generate(model, processor, images: list, prompts: list[str], max_new_tokens: int, num_beams: int,
             repetition_penalty: float) -> list[str]:
    texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = processor(images=[[im] for im in images], text=texts, add_special_tokens=False,
                       padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=num_beams,
                         repetition_penalty=repetition_penalty, use_cache=True)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in processor.batch_decode(new_tokens, skip_special_tokens=True)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="checkpoints/<run>/final_adapter, a checkpoint-N dir, or a base model "
                        "id (e.g. google/medgemma-1.5-4b-it, StanfordAIMI/CheXOne) for a zero-shot baseline")
    p.add_argument("--split", default=None, help="comma-separated manifest splits, e.g. test_chexpert,test_mimic")
    p.add_argument("--image_path", default=None, help="single image instead of a manifest split")
    p.add_argument("--view", default=None, help="view for --image_path: AP / PA / LATERAL")
    p.add_argument("--output_csv", default=None)
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    p.add_argument("--frontal_only", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="first N images only, for quick checks")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="default per model (rrg/models.py): 256 MedGemma, 352 Qwen-based = same word budget")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--save_every", type=int, default=25, help="flush CSV every N batches")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    spec = spec_for_adapter(args.adapter)
    args.max_new_tokens = args.max_new_tokens or spec.max_new_tokens
    print(f"[infer] base model: {spec.key}  image_size={spec.image_size or 'processor default'}"
          f"  max_new_tokens={args.max_new_tokens}")

    # use hints at inference iff the adapter was trained with them
    run_args_path = Path(args.adapter) / "run_args.json"
    run_args = json.loads(run_args_path.read_text()) if run_args_path.exists() else {}
    hints = None
    if run_args.get("hints"):
        hints = load_hints(run_args["hint_threshold"])
        print(f"[infer] adapter was trained with classifier hints (threshold {run_args['hint_threshold']})")

    model, processor = load_model(args.adapter, not args.no_4bit)

    if args.image_path:
        image = load_image(args.image_path, spec.image_size)
        hint = hints.get(args.image_path) if hints else None
        for task in args.tasks:
            text = generate(model, processor, [image], [build_prompt(task, args.view, hint)],
                            args.max_new_tokens, args.num_beams, args.repetition_penalty)[0]
            print(f"--- {task} ---\n{postprocess(text)}")
        return
    if not args.split:
        p.error("provide --split or --image_path")

    df = pd.read_parquet(MANIFEST_PATH)
    df = df[df["split"].isin(args.split.split(","))]
    if args.frontal_only:
        df = df[df["view"].ne("LATERAL")]
    if args.limit:
        df = df.head(args.limit)
    run_name = Path(args.adapter).parent.name if Path(args.adapter).name == "final_adapter" else Path(args.adapter).name
    out_csv = Path(args.output_csv or PRED_DIR / f"{run_name}__{args.split.replace(',', '+')}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.resume and out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
        done = {r["image_path"] for r in rows}
        df = df[~df["image_path"].isin(done)]
        print(f"[infer] resuming: {len(done):,} done, {len(df):,} to go")

    def flush():
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    records = df.to_dict("records")
    batches = [records[i:i + args.batch_size] for i in range(0, len(records), args.batch_size)]
    for b, batch in enumerate(tqdm(batches, desc="infer")):
        images = [load_image(r["image_path"], spec.image_size) for r in batch]
        result = [{"image_path": r["image_path"], "study_id": r["study_id"], "source": r["source"],
                   "split": r["split"], "view": r["view"],
                   "ground_truth_findings": r["findings"], "ground_truth_impression": r["impression"]}
                  for r in batch]
        for task in args.tasks:
            prompts = [build_prompt(task, r["view"], hints.get(r["image_path"]) if hints else None) for r in batch]
            texts = generate(model, processor, images, prompts, args.max_new_tokens, args.num_beams,
                             args.repetition_penalty)
            for res, text in zip(result, texts):
                res[f"generated_{task}_raw"] = text
                res[f"generated_{task}"] = postprocess(text)
        rows.extend(result)
        if (b + 1) % args.save_every == 0:
            flush()
    flush()
    print(f"[infer] {len(rows):,} rows -> {out_csv}")


if __name__ == "__main__":
    main()
