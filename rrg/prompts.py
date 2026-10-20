"""
Task prompts (Indonesian) and the optional classifier hints that can be added to them.
The same prompt text is used for every base model; only the chat template around it differs.
"""

from rrg.config import HINTS_PATH

TASK_INSTRUCTION = {
    "findings": "Tuliskan bagian TEMUAN dari laporan radiologi untuk foto toraks ini.",
    "impression": "Tuliskan bagian KESAN dari laporan radiologi untuk foto toraks ini.",
}
VIEW_TEXT = {"AP": "AP (portabel)", "PA": "PA", "LATERAL": "lateral", "LL": "lateral", "RL": "lateral"}

CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
LABEL_ID = {
    "Enlarged Cardiomediastinum": "pelebaran kardiomediastinum", "Cardiomegaly": "kardiomegali",
    "Lung Opacity": "opasitas paru", "Lung Lesion": "lesi paru", "Edema": "edema",
    "Consolidation": "konsolidasi", "Pneumonia": "pneumonia", "Atelectasis": "atelektasis",
    "Pneumothorax": "pneumotoraks", "Pleural Effusion": "efusi pleura", "Pleural Other": "kelainan pleura lain",
    "Fracture": "fraktur", "Support Devices": "alat medis terpasang", "No Finding": "tidak ada temuan",
}


def build_prompt(task: str, view: str | None = None, hint: str | None = None) -> str:
    lines = ["Anda adalah seorang ahli radiologi. " + TASK_INSTRUCTION[task]]
    if view and view in VIEW_TEXT:
        lines.append(f"Proyeksi: {VIEW_TEXT[view]}.")
    if hint:
        lines.append(f"Petunjuk dari model klasifikasi (bisa keliru): {hint}.")
    lines.append("Jawab dalam Bahasa Indonesia.")
    return "\n".join(lines)


def hint_from_probs(probs: dict, threshold: float) -> str:
    positives = [LABEL_ID[k] for k in CHEXPERT_LABELS if k != "No Finding" and probs.get(k, 0.0) >= threshold]
    return ", ".join(positives) if positives else LABEL_ID["No Finding"]


def load_hints(threshold: float) -> dict:
    """image_path -> hint text, from tools/export_classifier_hints.py's output."""
    import pandas as pd
    df = pd.read_parquet(HINTS_PATH)
    probs = df[CHEXPERT_LABELS].to_dict("records")
    return {path: hint_from_probs(pr, threshold) for path, pr in zip(df["image_path"], probs)}
