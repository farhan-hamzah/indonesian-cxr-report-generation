"""
Paths and split roles for the Indonesian CXR report-generation pipeline.

Split roles (mirrors ../rrg_pipeline so numbers stay comparable with the
English baseline):
  train_internal  CheXpert-Plus official train, patient-level 85% slice     -> training
  dev_internal    CheXpert-Plus official train, patient-level 15% slice     -> eval_loss / checkpoint selection
  train_mimic     MIMIC-CXR official validate (translated)                  -> training supplement
  test_chexpert   CheXpert-Plus official valid (234 images)                 -> final eval only
  test_mimic      MIMIC-CXR official test (translated)                      -> final eval only
  test_openi      OpenI / IU-Xray (translated)                              -> final eval only (external)
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # rrg_indo/
ROOT = Path(os.environ.get("RRG_ROOT", PROJECT_DIR.parent))
DATA_ROOT = Path(os.environ.get("RRG_DATA_ROOT", ROOT / "dataset"))
INDO_DIR = DATA_ROOT / "bahasaIndonesiaText"

CHEXPERT_INDO = INDO_DIR / "chexpert+" / "chexpert_plus_with_indonesian_full.parquet"
MIMIC_INDO = {"validate": INDO_DIR / "mimic" / "mimic_cxr_validate_translated.csv",
              "test": INDO_DIR / "mimic" / "mimic_cxr_test_translated.csv"}
OPENI_INDO = INDO_DIR / "openi" / "openi_translated_only.csv"

CHEXPERT_IMAGES = DATA_ROOT / "chexpert-plus" / "image-preprocess"
MIMIC_IMAGES = DATA_ROOT / "mimic-cxr" / "image-preprocess"
MIMIC_METADATA = DATA_ROOT / "mimic-cxr" / "label" / "mimic-cxr-2.0.0-metadata.parquet"
OPENI_IMAGES = DATA_ROOT / "openi-cxr" / "image-preprocess"

# patient -> train_internal/dev_internal assignment is reused from the English
# pipeline when available, so dev/test sets are identical across languages
REF_CHEXPERT_MANIFEST = ROOT / "rrg_pipeline" / "data" / "manifests" / "chexpert_manifest.parquet"
DEV_FRACTION = 0.15
SPLIT_SEED = 42

DATA_DIR = PROJECT_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.parquet"
HINTS_PATH = DATA_DIR / "classifier_hints.parquet"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"

TRAIN_SPLITS = ["train_internal", "train_mimic"]
DEV_SPLIT = "dev_internal"
TEST_SPLITS = ["test_chexpert", "test_mimic", "test_openi"]
TASKS = ["findings", "impression"]
