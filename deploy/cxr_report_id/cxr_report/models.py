"""
The models in this package and everything that differs between them. Each is a LoRA adapter
in models/<key>/ on top of a base model downloaded from Hugging Face; the prompts, image
preprocessing and post-processing are shared (the same as in training).

  medgemma   MedGemma 1.5 4B (Gemma 3) + LoRA   default; best on the external test set (OpenI)
  chexone    CheXOne 3B (Qwen2.5-VL) + LoRA     CC BY-NC 4.0: non-commercial use only

Settings are the ones behind the reported test metrics (rrg_indo/rrg/models.py, infer.py).
"""

import re
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    base_model: str  # Hugging Face id of the base model the adapter was trained on
    image_size: int | None  # square resize after the 224x224 step; None = the processor resizes (MedGemma: 896)
    max_new_tokens: int  # the same ~150-word budget for both (Qwen's tokenizer needs more tokens per word)
    license: str

    @property
    def adapter_dir(self) -> Path:
        return MODELS_DIR / self.key


MODELS = {spec.key: spec for spec in [
    ModelSpec("medgemma", "MedGemma 1.5 4B + LoRA (Bahasa Indonesia)", "google/medgemma-1.5-4b-it",
              image_size=None, max_new_tokens=256,
              license="Health AI Developer Foundations Terms of Use (see NOTICE.txt)"),
    ModelSpec("chexone", "CheXOne 3B + LoRA (Bahasa Indonesia)", "StanfordAIMI/CheXOne",
              image_size=448, max_new_tokens=352,
              license="CC BY-NC 4.0, non-commercial use only (see NOTICE.txt)"),
]}
DEFAULT_MODEL = "medgemma"

# Modules Unsloth keeps in 16-bit when it loads a base model in 4-bit for training
# (unsloth_zoo SKIP_QUANTIZATION_MODULES). Unsloth matches them as name components
# ("vision_tower" skips model.vision_tower.*); plain transformers 5.x only matches a prefix or
# suffix, so the names are turned into regexes with the same meaning. Result, checked layer by
# layer against Unsloth: MedGemma keeps its vision tower + lm_head in 16-bit, CheXOne its
# vision merger + lm_head.
SKIP_4BIT = ["lm_head", "multi_modal_projector", "merger", "modality_projection", "router", "mlp.gate",
             "block_sparse_moe.gate", "mamba", "audio_tower", "vision_tower", "vision_embedder", "embed_vision",
             "embed_audio", "score", "classifier", "qa_outputs"]
SKIP_4BIT_PATTERNS = [rf"(?:.*\.)?{re.escape(name)}(?:\..*)?$" for name in SKIP_4BIT]


def get_spec(model: str) -> ModelSpec:
    key = str(model).strip().lower()
    if key not in MODELS:
        raise ValueError(f"unknown model {model!r}; available: {', '.join(MODELS)}")
    return MODELS[key]
