"""
Pre-flight check of a base model (rrg/models.py) before spending GPU days on it. CPU only;
downloads the processor/tokenizer (a few MB), not the weights, unless --download.

  1. chat template    one training conversation rendered with the model's template: the
                      prompt-masking markers must each occur once, in order, right before the
                      target, and also as token ids (that's how Unsloth matches them)
  2. inference prompt infer.py's generation prompt ends with the same response marker
  3. image tokens     the real processor on one manifest image at the model's input size, and
                      on a 2-image batch exactly the way infer.py calls it
  4. stop token       the end-of-turn token after the target is one of generation_config's eos ids
  5. lengths          N random training samples: full sequence (image + prompt + target) vs
                      train.py --max_length, target tokens vs infer.py --max_new_tokens
  6. --download       fetch the weights into the HF cache so the GPU smoke test doesn't wait

    python tools/check_model.py --model chexone
    python tools/check_model.py --model chexone --download
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rrg_indo/, for the rrg package
from rrg.config import MANIFEST_PATH, TASKS  # noqa: E402
from rrg.data import chat_messages, load_image, train_dev_samples  # noqa: E402
from rrg.models import DEFAULT_MODEL, MODELS  # noqa: E402


def contains(seq: list[int], sub: list[int]) -> int:
    return sum(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def describe(x: np.ndarray, limit: int, name: str) -> str:
    over = int((x > limit).sum())
    return (f"{name}: p50={np.median(x):.0f} p95={np.percentile(x, 95):.0f} p99={np.percentile(x, 99):.0f} "
            f"max={x.max()}  > {limit}: {over}/{len(x)} ({over / len(x):.2%})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    p.add_argument("--n", type=int, default=500, help="training samples for the length statistics")
    p.add_argument("--max_length", type=int, default=1024, help="train.py --max_length")
    p.add_argument("--max_new_tokens", type=int, default=None, help="default: the model's infer.py default")
    p.add_argument("--mimic_repeat", type=int, default=4)
    p.add_argument("--download", action="store_true", help="also download the weights (~8 GB for CheXOne)")
    args = p.parse_args()
    spec = MODELS[args.model]
    args.max_new_tokens = args.max_new_tokens or spec.max_new_tokens
    print(f"[check] {spec.key}: {spec.hf_id}\n        {spec.description}")

    from transformers import AutoProcessor, GenerationConfig
    processor = AutoProcessor.from_pretrained(spec.hf_id)
    tok = processor.tokenizer
    tok.padding_side = "left"  # as infer.py
    train, _, _ = train_dev_samples(pd.read_parquet(MANIFEST_PATH), TASKS, mimic_repeat=args.mimic_repeat)
    checks = {}

    # 1. chat template + masking markers
    s = train[0]
    image = load_image(s["image_path"], spec.image_size)
    text = processor.apply_chat_template(chat_messages(image, s["prompt"], s["target"]), tokenize=False)
    print(f"\n--- rendered training sample ---\n{text}--- end ---\n")
    i_pos, r_pos = text.find(spec.instruction_part), text.find(spec.response_part)
    checks["instruction_part occurs once"] = text.count(spec.instruction_part) == 1
    checks["response_part occurs once"] = text.count(spec.response_part) == 1
    checks["user turn comes before model turn"] = 0 <= i_pos < r_pos
    checks["target starts right after response_part"] = text[r_pos + len(spec.response_part):].startswith(s["target"])
    ids = tok(text, add_special_tokens=False).input_ids
    for name in ("instruction_part", "response_part"):
        marker = tok(getattr(spec, name), add_special_tokens=False).input_ids
        checks[f"{name} found once as token ids {marker}"] = contains(ids, marker) == 1

    # 2. inference prompt (infer.py.generate)
    gen_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s["prompt"]}]}],
        add_generation_prompt=True)
    checks["generation prompt ends with response_part"] = gen_text.endswith(spec.response_part)

    # 3. image tokens, single image and a 2-image batch as infer.py builds it
    image_token_id = tok.convert_tokens_to_ids(processor.image_token)
    s2 = train[1]
    gen_text2 = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s2["prompt"] + " (contoh)"}]}],
        add_generation_prompt=True)
    batch = processor(images=[[image], [load_image(s2["image_path"], spec.image_size)]], text=[gen_text, gen_text2],
                      add_special_tokens=False, padding=True, return_tensors="pt")
    per_row = (batch["input_ids"] == image_token_id).sum(dim=1).tolist()
    print(f"[check] image {image.size[0]}x{image.size[1]} -> {per_row[0]} image tokens; "
          f"batch input_ids {tuple(batch['input_ids'].shape)}, image tokens per row {per_row}")
    checks[f"{spec.image_tokens} image tokens per image (batched)"] = per_row == [spec.image_tokens] * 2

    # 4. stop token
    after_target = text[text.find(s["target"]) + len(s["target"]):]
    end_id = tok(after_target, add_special_tokens=False).input_ids[0]
    eos = GenerationConfig.from_pretrained(spec.hf_id).eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    checks[f"end-of-turn {tok.convert_ids_to_tokens(end_id)!r} is a generation eos id {eos}"] = end_id in eos

    # 5. lengths
    rng = random.Random(0)
    sample = rng.sample(train, min(args.n, len(train)))
    seq, tgt = [], []
    for x in sample:
        msgs = chat_messages(load_image(x["image_path"], spec.image_size), x["prompt"], x["target"])
        full = processor.apply_chat_template(msgs, tokenize=False)
        seq.append(processor(images=[[msgs[0]["content"][0]["image"]]], text=[full], add_special_tokens=False,
                             return_tensors="pt")["input_ids"].shape[1])
        tgt.append(len(tok(x["target"], add_special_tokens=False).input_ids))
    seq, tgt = np.array(seq), np.array(tgt)
    words = np.array([len(x["target"].split()) for x in sample])
    print(f"[check] {len(sample)} random training samples (tokens per word {tgt.sum() / words.sum():.2f})")
    print("        " + describe(seq, args.max_length, "full sequence"))
    print("        " + describe(tgt, args.max_new_tokens, "target only  "))

    print()
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in checks.items() if not ok]

    if args.download:
        from huggingface_hub import snapshot_download
        print(f"\n[check] downloading {spec.hf_id} weights ...")
        print(f"[check] -> {snapshot_download(spec.hf_id)}")

    if failed:
        print(f"\n[check] {len(failed)} check(s) failed: fix rrg/models.py before training {spec.key}")
        sys.exit(1)
    print(f"\n[check] {spec.key} is ready. GPU smoke test:\n"
          f"  python train.py --model {spec.key} --run_name {spec.key}_smoke --max_train_samples 64 --max_steps 5 "
          f"--eval_samples 16 --eval_steps 5 --freeze_vision\n"
          f"  python infer.py --adapter checkpoints/{spec.key}_smoke/final_adapter --split test_chexpert --limit 8")


if __name__ == "__main__":
    main()
