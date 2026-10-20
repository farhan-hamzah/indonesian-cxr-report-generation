"""
Verbatim-copy (memorization) check on test_mimic, CPU only. Is a generated report identical to a
training report more often than the test REFERENCES themselves are (base rate)? Also the share of
the outputs' 8-grams that occur in MIMIC training targets, against the same share for the references.
Written to check whether --mimic_repeat 4 made the models memorize the MIMIC training reports.

    uv run python tools/check_memorization.py [run ...]      (default: medgemma_full chexone_full)
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rrg.config import MANIFEST_PATH  # noqa: E402
from rrg.text import clean_target  # noqa: E402

runs = sys.argv[1:] or ["medgemma_full", "chexone_full"]
m = pd.read_parquet(MANIFEST_PATH)


def norm(text) -> str:
    return " ".join(str(text).lower().replace(".", " .").replace(",", " ,").split())


def grams(text: str, n: int = 8) -> set:
    w = text.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def gram_share(texts, pool: set) -> float:
    g = [grams(t) for t in texts]
    total = sum(len(x) for x in g)
    return sum(len(x & pool) for x in g) / max(total, 1)


for task in ["findings", "impression"]:
    tm = m[m.split.eq("train_mimic")]
    targets = tm[task].map(clean_target)
    train_mimic = targets.dropna().map(norm)
    train_chexpert = set(m[m.split.eq("train_internal")][task].map(clean_target).dropna().map(norm))
    set_mimic = set(train_mimic)
    # a study's report repeats over its images, so count training STUDIES per text:
    # copying a text written for a single study = reproducing one patient's report
    n_studies = (pd.DataFrame({"text": targets.map(norm, na_action="ignore"), "study": tm.study_id})
                 .dropna().groupby("text")["study"].nunique().to_dict())
    grams_mimic = set().union(*[grams(t) for t in set_mimic])
    test = m[m.split.eq("test_mimic") & m[task].notna()]
    refs = test[task].map(norm)
    print(f"\n=== test_mimic {task}  (MIMIC train targets {len(train_mimic):,}, {len(set_mimic):,} unique)")
    single = sum(v == 1 for v in n_studies.values())
    print(f"  ({single:,} of {len(n_studies):,} distinct MIMIC training texts were written for a single study)")
    print(f"  {'references (base rate)':<24} == 1-study report: {refs.map(lambda t: n_studies.get(t, 0) == 1).mean():6.1%}"
          f"   == MIMIC train: {refs.isin(set_mimic).mean():6.1%}"
          f"   == CheXpert train: {refs.isin(train_chexpert).mean():6.1%}"
          f"   8-grams in MIMIC train: {gram_share(refs, grams_mimic):6.1%}")
    for run in runs:
        p = pd.read_csv(ROOT / "outputs" / run / "predictions" / f"{run}__test_mimic.csv")
        gen = p[p.image_path.isin(test.image_path)][f"generated_{task}"].fillna("").map(norm)
        in_mimic = gen.isin(set_mimic)
        print(f"  {run:<24} == 1-study report: {gen.map(lambda t: n_studies.get(t, 0) == 1).mean():6.1%}"
              f"   == MIMIC train: {in_mimic.mean():6.1%}"
              f"   == CheXpert train: {gen.isin(train_chexpert).mean():6.1%}"
              f"   8-grams in MIMIC train: {gram_share(gen, grams_mimic):6.1%}"
              f"   ({gen[in_mimic].nunique()} distinct copied texts)")
        top = Counter(gen[in_mimic]).most_common(1)
        if top:
            print(f"      most frequent copy ({top[0][1]}x): {top[0][0][:100]}")
