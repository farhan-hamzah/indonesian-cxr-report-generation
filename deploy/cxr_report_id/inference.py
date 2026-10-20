"""
Command-line report generation.

    python inference.py --image cxr.png --view PA                       # MedGemma (default)
    python inference.py --image cxr.png --view PA --model chexone
    python inference.py --image cxr.png --view PA --model all           # every model, one after another
    python inference.py --image a.png b.png --view PA LATERAL --json
    python inference.py --image cxr.png --cpu                           # no GPU: slow (minutes per image)
"""

import argparse
import gc
import json
import time

import torch

from cxr_report import MODELS, ReportGenerator

SECTION_TITLE = {"findings": "TEMUAN", "impression": "KESAN"}


def main():
    p = argparse.ArgumentParser(description="Indonesian chest X-ray report (TEMUAN + KESAN)")
    p.add_argument("--image", nargs="+", required=True, help="PNG/JPEG file(s); DICOM must be exported first")
    p.add_argument("--view", nargs="+", default=None,
                   help="AP / PA / LATERAL per image (one value is used for all); omit if unknown")
    p.add_argument("--model", default="medgemma", choices=[*MODELS, "all"])
    p.add_argument("--base_model", default=None, help="local base-model folder instead of the HF download "
                                                       "(only with a single --model)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_4bit", action="store_true", help="16-bit weights on GPU (needs more VRAM, see README)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    p.add_argument("--raw", action="store_true", help="also return the output before post-processing")
    args = p.parse_args()

    views = args.view or [None]
    if len(views) == 1:
        views = views * len(args.image)
    if len(views) != len(args.image):
        p.error(f"{len(args.image)} images but {len(views)} views")
    keys = list(MODELS) if args.model == "all" else [args.model]
    if args.base_model and len(keys) > 1:
        p.error("--base_model needs a single --model")

    results = {}  # model -> list of reports
    for key in keys:
        t0 = time.time()
        gen = ReportGenerator(key, base_model=args.base_model, device="cpu" if args.cpu else None,
                              load_in_4bit=False if (args.cpu or args.no_4bit) else None)
        print(f"[{key}] loaded in {time.time() - t0:.0f} s: {gen.info}", flush=True)
        results[key] = []
        for i in range(0, len(args.image), args.batch_size):
            t0 = time.time()
            batch = gen.generate_batch(args.image[i:i + args.batch_size], views[i:i + args.batch_size],
                                       return_raw=args.raw)
            print(f"[{key}] {len(batch)} image(s) in {time.time() - t0:.1f} s", flush=True)
            results[key].extend(batch)
        del gen  # free the GPU before the next model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.json:
        out = [{"image": path, "view": view, "reports": {key: results[key][i] for key in keys}}
               for i, (path, view) in enumerate(zip(args.image, views))]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    for i, (path, view) in enumerate(zip(args.image, views)):
        print(f"\n=== {path} (proyeksi: {view or 'tidak diketahui'}) ===")
        for key in keys:
            if len(keys) > 1:
                print(f"--- {MODELS[key].name} ---")
            for task, title in SECTION_TITLE.items():
                print(f"{title}: {results[key][i][task]}")


if __name__ == "__main__":
    main()
