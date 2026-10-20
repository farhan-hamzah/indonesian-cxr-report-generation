"""
Self-test on the machine that will run the app: environment, file integrity, and per model the load
time, VRAM and seconds per image. Run it once after installation (README "Instalasi").

    python selftest.py                              # both models, synthetic test image
    python selftest.py --image cxr.png --view PA    # a real chest X-ray: also prints the two reports
    python selftest.py --model medgemma             # one model only
    python selftest.py --cpu                        # no GPU (slow)

With the synthetic image (no X-ray is shipped with this folder) the reports are meaningless; the test
only proves that loading and generation work. Use --image with a real chest X-ray to judge the text.
"""

import argparse
import hashlib
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHA256 = {  # adapter_model.safetensors, as listed in README "Versi"
    "medgemma": "a387a1f0a0cd00a386023347e704c66b7dbb16fc3b82a745001ce7f99bf5487e",
    "chexone": "9c113d125061abcfdcb688aae9d69dc4cf09d648a03bf819af5bc0e6d3637fbc",
}


def ok(flag: bool) -> str:
    return "OK  " if flag else "FAIL"


def synthetic_cxr() -> np.ndarray:
    """A 224x224 grayscale image with roughly chest-like contrast (dark lungs, bright mediastinum)."""
    y, x = np.mgrid[0:224, 0:224] / 223.0
    img = 0.25 + 0.5 * np.exp(-((x - 0.5) / 0.12) ** 2) + 0.2 * y
    for cx in (0.3, 0.7):
        img -= 0.25 * np.exp(-(((x - cx) / 0.15) ** 2 + ((y - 0.45) / 0.3) ** 2))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="environment + model self-test")
    p.add_argument("--model", default="all", choices=["medgemma", "chexone", "all"])
    p.add_argument("--image", default=None, help="a real chest X-ray (PNG/JPEG); default: synthetic image")
    p.add_argument("--view", default=None, help="AP / PA / LATERAL for --image")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    failures = 0

    print("== environment")
    import torch
    import transformers
    import peft
    print(f"python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__} "
          f"| peft {peft.__version__}")
    gpu = torch.cuda.is_available() and not args.cpu
    if gpu:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU {props.name}, {props.total_memory / 2**30:.1f} GiB, bfloat16 {torch.cuda.is_bf16_supported()}")
        try:
            import bitsandbytes
            print(f"bitsandbytes {bitsandbytes.__version__}")
        except ImportError:
            print("FAIL bitsandbytes is not installed (needed for 4-bit on GPU): pip install -r requirements.txt")
            failures += 1
    else:
        print("no GPU in use: CPU mode (slow, ~10 GB RAM per model)")
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        print("note: tested with Python 3.11 / 3.12")

    from cxr_report import MODELS, ReportGenerator
    keys = list(MODELS) if args.model == "all" else [args.model]

    print("\n== files")
    for key in keys:
        f = MODELS[key].adapter_dir / "adapter_model.safetensors"
        if not f.exists():
            print(f"FAIL {key}: {f} missing")
            failures += 1
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        good = h == SHA256[key]
        failures += not good
        print(f"{ok(good)} {key}: adapter sha256 {h[:16]}...{'' if good else ' (expected ' + SHA256[key][:16] + '...: re-copy the folder)'}")

    image = args.image or synthetic_cxr()
    print(f"\n== models (image: {args.image or 'synthetic'})")
    rows = []
    for key in keys:
        if gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            gen = ReportGenerator(key, device="cpu" if not gpu else None)
            load_s = time.time() - t0
            vram_load = torch.cuda.memory_allocated() / 2**30 if gpu else float("nan")
            t0 = time.time()
            rep = gen.generate(image, view=args.view)
            first_s = time.time() - t0
            t0 = time.time()
            gen.generate(image, view=args.view)
            warm_s = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30 if gpu else float("nan")
            good = isinstance(rep.get("findings"), str) and isinstance(rep.get("impression"), str)
            failures += not good
            rows.append((key, load_s, vram_load, peak, first_s, warm_s))
            print(f"{ok(good)} {key}: {gen.info}")
            if args.image:
                print(f"     TEMUAN: {rep['findings']}\n     KESAN : {rep['impression']}")
            del gen
        except Exception as e:  # report and continue with the next model
            failures += 1
            print(f"FAIL {key}: {type(e).__name__}: {e}")
        if gpu:
            torch.cuda.empty_cache()

    if rows:
        print(f"\n{'model':<9} {'load s':>7} {'VRAM loaded GiB':>16} {'VRAM peak GiB':>14} {'1st image s':>12} {'next image s':>13}")
        for key, a, b, c, d, e in rows:
            print(f"{key:<9} {a:>7.0f} {b:>16.1f} {c:>14.1f} {d:>12.1f} {e:>13.1f}")
    print("\nSELFTEST " + ("PASSED" if failures == 0 else f"FAILED ({failures} problem(s) above)"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
