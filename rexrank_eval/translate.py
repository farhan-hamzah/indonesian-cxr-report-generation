"""
Indonesian -> English translation of report texts, so English-only ReXrank
metrics (SembScore/CheXbert, RadGraph, RadCliQ, RaTEScore, GREEN, CRIMSON) can
score Indonesian model output against the ORIGINAL English reference reports.

Translator: google/medgemma-1.5-4b-it (base model, NOT the fine-tuned adapter;
already in the local HF cache), greedy decoding. Every unique text is translated
once and cached in outputs/translation_cache/, shared by all runs, so re-running
on another prediction CSV only translates what's new. The Indonesian ground-truth
columns are translated too: scoring those back-translations against the English
originals gives the translation-noise ceiling (see rexrank.py --calibration).

    uv run python translate.py --csv ../outputs/medgemma_full/predictions/medgemma_full__test_mimic.csv
"""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE.parent / "outputs" / "translation_cache"
TRANSLATOR = "google/medgemma-1.5-4b-it"
COLUMNS = ["generated_findings", "generated_impression", "ground_truth_findings", "ground_truth_impression"]

PROMPT = (
    "Translate this Indonesian chest X-ray report text into English. Use standard English radiology "
    "terminology. Translate faithfully: keep every finding, negation, laterality, size and list numbering; "
    "do not add, remove or explain anything. Output only the English translation.\n\n{text}"
)
LEADING_LABEL = re.compile(r"^\s*(?:\*\*)?(?:english(?: translation)?|translation)\s*:?\s*(?:\*\*)?\s*:?\s*", re.I)


def cache_path(translator: str) -> Path:
    return CACHE_DIR / f"id2en__{translator.split('/')[-1]}.parquet"


def load_cache(translator: str = TRANSLATOR) -> dict:
    p = cache_path(translator)
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return dict(zip(df["id"], df["en"]))


def save_cache(cache: dict, translator: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(translator)
    tmp = p.with_suffix(".tmp")
    pd.DataFrame({"id": list(cache), "en": list(cache.values())}).to_parquet(tmp, index=False)
    tmp.replace(p)


def clean_output(text: str) -> str:
    t = LEADING_LABEL.sub("", text.strip())
    t = t.strip().strip('"').strip()
    return re.sub(r"\s+", " ", t)


def collect_texts(csvs: list[str], columns: list[str]) -> list[str]:
    texts = set()
    for c in csvs:
        df = pd.read_csv(c)
        for col in columns:
            if col in df:
                texts.update(t.strip() for t in df[col].dropna().astype(str) if t.strip())
    return sorted(texts)


def max_new_tokens(src_len: int) -> int:
    return min(1024, 2 * src_len + 48)


@torch.inference_mode()
def translate_all(texts: list[str], translator: str, batch_size: int, token_budget: int, save_every: int, cache: dict):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(translator)
    tok = processor.tokenizer
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        translator, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    def prompt(t):
        return tok.apply_chat_template([{"role": "user", "content": PROMPT.format(text=t)}],
                                       tokenize=False, add_generation_prompt=True)

    prompt_len = len(tok(prompt(""), add_special_tokens=False)["input_ids"])
    # longest first: an OOM shows up early, and similar lengths pad less
    lens = {t: len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts}
    texts = sorted(texts, key=lambda t: lens[t], reverse=True)
    i, done_since_save, start = 0, 0, time.time()
    while i < len(texts):
        # batch size from a token budget (KV cache ~ batch x (prompt + source + output)), so a handful
        # of very long texts doesn't shrink the batch for the thousands of short ones
        src_len = lens[texts[i]]
        bs = max(1, min(batch_size, token_budget // (prompt_len + src_len + max_new_tokens(src_len))))
        batch = texts[i:i + bs]
        enc = tok([prompt(t) for t in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        try:
            out = model.generate(**enc, max_new_tokens=max_new_tokens(src_len), do_sample=False)
        except torch.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            token_budget //= 2
            print(f"[translate] OOM at batch {bs} -> token budget {token_budget}", flush=True)
            continue
        new = out[:, enc["input_ids"].shape[1]:]
        for src, en in zip(batch, tok.batch_decode(new, skip_special_tokens=True)):
            cache[src] = clean_output(en)
        i += len(batch)
        done_since_save += 1
        if done_since_save >= save_every:
            save_cache(cache, translator)
            done_since_save = 0
            rate = i / (time.time() - start)
            print(f"[translate] {i:,}/{len(texts):,} ({rate:.1f} texts/s, ~{(len(texts) - i) / rate / 60:.0f} min left)",
                  flush=True)
    save_cache(cache, translator)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True, help="prediction CSVs from infer.py / tools/postprocess_csv.py")
    p.add_argument("--columns", nargs="+", default=COLUMNS)
    p.add_argument("--translator", default=TRANSLATOR)
    p.add_argument("--batch_size", type=int, default=64, help="max texts per batch")
    p.add_argument("--token_budget", type=int, default=24576,
                   help="max batch x (prompt+source+output) tokens; halved on OOM")
    p.add_argument("--save_every", type=int, default=10, help="batches between cache writes")
    p.add_argument("--limit", type=int, default=None, help="translate only N new texts (smoke test)")
    args = p.parse_args()

    cache = load_cache(args.translator)
    texts = [t for t in collect_texts(args.csv, args.columns) if t not in cache]
    if args.limit:
        texts = texts[:args.limit]
    print(f"[translate] {len(cache):,} cached, {len(texts):,} new unique texts", flush=True)
    if texts:
        translate_all(texts, args.translator, args.batch_size, args.token_budget, args.save_every, cache)
    print(f"[translate] cache -> {cache_path(args.translator)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
