"""
Adds post-processed columns (rrg/postprocess.py) to a prediction CSV written before
infer.py did it itself: generated_<task>_raw = original output, generated_<task> = cleaned.

    python tools/postprocess_csv.py --predictions_csv outputs/predictions/medgemma_40k__test_openi.csv \
        --output_csv outputs/medgemma_40k/predictions/medgemma_40k__test_openi.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rrg_indo/, for the rrg package
from rrg.postprocess import postprocess  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--output_csv", required=True, help="new file; the input CSV is left as-is")
    args = p.parse_args()

    df = pd.read_csv(args.predictions_csv)
    for task in ("findings", "impression"):
        col = f"generated_{task}"
        if col not in df:
            continue
        raw_col = f"{col}_raw"
        if raw_col not in df:
            df[raw_col] = df[col]
        df[col] = df[raw_col].map(postprocess)
        changed = (df[col].fillna("") != df[raw_col].fillna("").str.strip()).mean()
        print(f"[postprocess] {col}: changed {changed:.1%} of {len(df):,} rows")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[postprocess] -> {args.output_csv}")


if __name__ == "__main__":
    main()
