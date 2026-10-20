"""
Builds data/manifest.parquet: one row per image with cleaned Indonesian
findings/impression (rrg.text.clean_text), view position and split role
(see rrg/config.py). Only rows whose image exists on disk are kept.
The manifest is model-independent: every base model trains on it.

    python prepare_data.py
"""

import os
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from rrg.config import (
    CHEXPERT_INDO, MIMIC_INDO, OPENI_INDO, CHEXPERT_IMAGES, MIMIC_IMAGES, MIMIC_METADATA, OPENI_IMAGES,
    REF_CHEXPERT_MANIFEST, DEV_FRACTION, SPLIT_SEED, DATA_DIR, MANIFEST_PATH,
)
from rrg.text import clean_text

COLUMNS = ["image_path", "study_id", "patient_id", "source", "split", "view", "findings", "impression"]

# translated_target_report_text is "temuan: <findings> kesan: <impression>" (either part optional)
_REPORT_RE = re.compile(r"^\s*(?:temuan\s*:\s*(?P<f>.*?))?\s*(?:\bkesan\s*:\s*(?P<i>.*))?$", re.IGNORECASE | re.DOTALL)


def _has_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _keep_existing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    exists = df["image_path"].map(os.path.exists)
    if (~exists).any():
        print(f"[{name}] dropped {(~exists).sum():,}/{len(df):,} rows with no image on disk")
    return df[exists]


def build_chexpert() -> pd.DataFrame:
    df = pd.read_parquet(CHEXPERT_INDO)
    parts = df["translated_target_report_text"].fillna("").str.extract(_REPORT_RE)
    # trust a translated section only where the English source has that section,
    # which drops the ~1k rows whose translation lost its temuan/kesan structure
    df["findings"] = parts["f"].where(_has_text(df["section_findings"])).map(clean_text)
    df["impression"] = parts["i"].where(_has_text(df["section_impression"])).map(clean_text)

    rel = df["path_to_image"].str.replace("\\", "/", regex=False)
    df["image_path"] = rel.map(lambda p: str(CHEXPERT_IMAGES / p).rsplit(".", 1)[0] + ".png")
    df["study_id"] = rel.str.rsplit("/", n=1).str[0]
    df["patient_id"] = df["deid_patient_id"]
    df["source"] = "chexpert_plus"
    df["view"] = df["ap_pa"].str.upper().where(df["frontal_lateral"].ne("Lateral"), "LATERAL")

    train = df["split"].eq("train")
    if REF_CHEXPERT_MANIFEST.exists():
        ref = pd.read_parquet(REF_CHEXPERT_MANIFEST, columns=["patient_id", "split"]).drop_duplicates("patient_id")
        patient_split = dict(zip(ref["patient_id"], ref["split"]))
        df.loc[train, "split"] = df.loc[train, "patient_id"].map(patient_split)
        print(f"[chexpert] reused patient split from {REF_CHEXPERT_MANIFEST}")
        df = df[df["split"].notna() | ~train]
    else:
        print("[chexpert] reference manifest not found, making a fresh patient-level split")
        sub = df[train]
        splitter = GroupShuffleSplit(n_splits=1, test_size=DEV_FRACTION, random_state=SPLIT_SEED)
        _, dev_idx = next(splitter.split(sub, groups=sub["patient_id"]))
        dev_patients = set(sub["patient_id"].iloc[dev_idx])
        df.loc[train, "split"] = df.loc[train, "patient_id"].isin(dev_patients).map(
            {True: "dev_internal", False: "train_internal"})
    df.loc[df["split"].eq("valid"), "split"] = "test_chexpert"
    return _keep_existing(df[COLUMNS], "chexpert")


def build_mimic() -> pd.DataFrame:
    meta = pd.read_parquet(MIMIC_METADATA, columns=["dicom_id", "subject_id", "study_id", "ViewPosition"])
    frames = []
    for official_split, path in MIMIC_INDO.items():
        rep = pd.read_csv(path, usecols=["subject_id", "study_id", "findings_id", "impression_id"])
        rep["split"] = "train_mimic" if official_split == "validate" else "test_mimic"
        frames.append(rep)
    df = pd.concat(frames).merge(meta, on=["subject_id", "study_id"], how="inner")

    df["findings"] = df["findings_id"].map(clean_text)
    df["impression"] = df["impression_id"].map(clean_text)
    df["image_path"] = [
        str(MIMIC_IMAGES / f"p{str(s)[:2]}" / f"p{s}" / f"s{st}" / f"{d}.png")
        for s, st, d in zip(df["subject_id"], df["study_id"], df["dicom_id"])
    ]
    df["patient_id"] = df["subject_id"].astype(str)
    df["study_id"] = df["subject_id"].astype(str) + "_" + df["study_id"].astype(str)
    df["source"] = "mimic_cxr"
    df["view"] = df["ViewPosition"].fillna("").astype(str).str.upper()
    return _keep_existing(df[COLUMNS], "mimic")


def build_openi() -> pd.DataFrame:
    rep = pd.read_csv(OPENI_INDO, usecols=["uid", "image_ids", "findings_id", "impression_id"])
    rep = rep[rep["image_ids"].notna()]
    rep = rep.assign(image_id=rep["image_ids"].str.split("|")).explode("image_id")
    rep = rep[rep["image_id"].fillna("").str.len() > 0]
    df = pd.DataFrame({
        "image_path": rep["image_id"].map(lambda i: str(OPENI_IMAGES / f"{i}.png")),
        "study_id": rep["uid"], "patient_id": rep["uid"], "source": "openi_cxr", "split": "test_openi",
        "view": "",  # OpenI ships no view metadata
        "findings": rep["findings_id"].map(clean_text), "impression": rep["impression_id"].map(clean_text),
    })
    return _keep_existing(df[COLUMNS], "openi")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([build_chexpert(), build_mimic(), build_openi()], ignore_index=True)
    df = df[df["findings"].notna() | df["impression"].notna()].reset_index(drop=True)
    df.to_parquet(MANIFEST_PATH, index=False)

    summary = df.groupby("split").agg(
        images=("image_path", "size"), studies=("study_id", "nunique"),
        findings=("findings", lambda s: s.notna().sum()), impression=("impression", lambda s: s.notna().sum()))
    print(summary.to_string())
    for task in ("findings", "impression"):
        words = df[task].dropna().str.split().str.len()
        print(f"{task} words: p50={words.quantile(.5):.0f} p95={words.quantile(.95):.0f} p99={words.quantile(.99):.0f}")
    print(f"saved -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
