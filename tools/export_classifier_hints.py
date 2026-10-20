"""
OPTIONAL. Scores every manifest image with the existing stage-2 pathology
classifier from ../rrg_pipeline (frozen BioViL-T + MLP head, trained on
CheXpert labels - image-only, so it is language-independent and needs no
retraining for Indonesian). train.py/infer.py --hints then put the predicted
positive labels into the prompt (RaDialog-style).

Must run inside the rrg_pipeline environment (needs hi-ml-multimodal), not
this folder's Unsloth environment, so it doesn't import the rrg package:

    cd ../rrg_pipeline
    uv run python ../rrg_indo/tools/export_classifier_hints.py

Caveat: the classifier was trained on train_internal, so its hints are more
accurate on training images than on test images; the generator may learn to
over-trust them. Compare against a run without --hints before relying on it.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # rrg_indo/
sys.path.insert(0, str(PROJECT_DIR.parent / "rrg_pipeline"))  # for backbones.py

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from backbones import BiovilTEncoder  # noqa: E402

CKPT = PROJECT_DIR.parent / "rrg_pipeline" / "checkpoints" / "stage2_classifier" / "best.pt"
MANIFEST_PATH = PROJECT_DIR / "data" / "manifest.parquet"
HINTS_PATH = PROJECT_DIR / "data" / "classifier_hints.parquet"
CHEXPERT_LABELS = [  # same order as rrg/prompts.py
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
BATCH_SIZE = 64


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BiovilTEncoder().to(device).eval()

    state = torch.load(CKPT, map_location=device)["head_state_dict"]
    hidden_dim, embed_dim = state["0.weight"].shape  # same layout as stage2_classifier.PathologyClassifier.head
    head = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
                         nn.Linear(hidden_dim, len(CHEXPERT_LABELS))).to(device).eval()
    head.load_state_dict(state)

    paths = pd.read_parquet(MANIFEST_PATH, columns=["image_path"])["image_path"].unique().tolist()
    probs = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="classifier hints"):
        batch = torch.stack([encoder.transform(Image.open(p).convert("L")) for p in paths[i:i + BATCH_SIZE]])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = head(encoder.forward_pooled(batch.to(device)))
        probs.append(torch.sigmoid(logits.float()).cpu())

    out = pd.DataFrame(torch.cat(probs).numpy(), columns=CHEXPERT_LABELS)
    out.insert(0, "image_path", paths)
    out.to_parquet(HINTS_PATH, index=False)
    print(f"saved {len(out):,} rows -> {HINTS_PATH}")


if __name__ == "__main__":
    main()
