"""
Manifest -> training samples, and image loading shared by train.py and infer.py.

One sample = one image + one task: each image yields a "findings" sample and/or an
"impression" sample (whichever references exist), told apart by the prompt. The sample
list depends only on the manifest and the data flags (not on the model), so every base
model is trained on the same samples in the same order.
"""

import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from rrg.config import DEV_SPLIT, TRAIN_SPLITS
from rrg.prompts import build_prompt
from rrg.text import clean_target


def load_image(path: str, size: int | None = None) -> Image.Image:
    """RGB image, resized to size x size when the model needs a fixed input size (manifest images are square)."""
    image = Image.open(path).convert("RGB")
    if size and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def build_samples(df: pd.DataFrame, tasks, hints: dict | None, clean: bool = True) -> list[dict]:
    samples = []
    for row in df.itertuples(index=False):
        for task in tasks:
            target = getattr(row, task)
            if clean:
                target = clean_target(target)
            if isinstance(target, str) and target:
                hint = hints.get(row.image_path) if hints is not None else None
                samples.append({"image_path": row.image_path, "prompt": build_prompt(task, row.view, hint),
                                "target": target})
    return samples


def train_dev_samples(manifest: pd.DataFrame, tasks, hints: dict | None = None, *, clean: bool = True,
                      mimic_repeat: int = 1, frontal_only: bool = False, max_train_samples: int | None = None,
                      eval_samples: int = 256, seed: int = 3407) -> tuple[list[dict], list[dict], int]:
    """(shuffled train samples, fixed dev subset for eval_loss, MIMIC samples per repeat)."""
    train_df = manifest[manifest["split"].isin(TRAIN_SPLITS)]
    if frontal_only:
        train_df = train_df[train_df["view"].ne("LATERAL")]
    train = build_samples(train_df, tasks, hints, clean)
    mimic = build_samples(train_df[train_df["source"].eq("mimic_cxr")], tasks, hints, clean)
    train += mimic * (mimic_repeat - 1)  # MIMIC is ~2% of the data
    dev = build_samples(manifest[manifest["split"].eq(DEV_SPLIT)], tasks, hints, clean)
    random.Random(seed).shuffle(train)
    if max_train_samples:
        train = train[:max_train_samples]
    dev = random.Random(0).sample(dev, min(eval_samples, len(dev)))
    return train, dev, len(mimic)


def chat_messages(image, prompt: str, target: str | None = None) -> list[dict]:
    """User turn (image + prompt), plus the assistant turn when a target is given."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


class ReportSFTDataset(Dataset):
    """Loads images lazily so 200k+ samples never sit in RAM at once."""

    def __init__(self, samples: list[dict], image_size: int | None = None):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {"messages": chat_messages(load_image(s["image_path"], self.image_size), s["prompt"], s["target"])}
