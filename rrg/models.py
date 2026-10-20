"""
Base vision-language models the pipeline can fine-tune, and everything that differs
between them. train.py / infer.py are model-agnostic; adding a model = one entry here
(then check it with tools/check_model.py).

  medgemma       google/medgemma-1.5-4b-it    Gemma 3 4B, medical generalist, multilingual
  chexone        StanfordAIMI/CheXOne         Qwen2.5-VL-3B post-trained on 14.7M CXR samples
                                              (MIMIC-CXR, CheXpert Plus, ...; IU-Xray held out),
                                              English-only, CC BY-NC 4.0 (research use only)
  qwen2.5-vl-3b  Qwen/Qwen2.5-VL-3B-Instruct  CheXOne's general-domain base: optional control that
                                              isolates the effect of CheXOne's CXR post-training

Image budget: every model sees 256 visual tokens per image. MedGemma's processor upsamples
our 224x224 PNGs to 896x896 itself (SigLIP 64x64 patches, pooled to 256). Qwen2.5-VL is
dynamic-resolution and would keep 224x224 (only 64 tokens), so images are resized to 448x448
first: (448/14)^2 patches / 2x2 merge = 256 tokens, within CheXOne's 512x512 training size.

Text budget: the same ~154 Indonesian words of output for every model. Qwen's tokenizer
needs 2.25 tokens per Indonesian word vs Gemma's 1.66 (tools/check_model.py, 500 training
targets), so infer.py's default max_new_tokens is 256 for MedGemma (as in medgemma_full)
and 256 * 2.25 / 1.66 ~ 352 for the Qwen-based models.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    instruction_part: str  # chat-template text that opens a user turn (prompt masking)
    response_part: str  # chat-template text that opens the model's turn; the loss covers what follows
    image_size: int | None  # square resize before the processor; None = leave it to the processor
    image_tokens: int  # visual tokens per image at that size
    max_new_tokens: int  # infer.py default generation budget (same word budget for every model)
    description: str


_GEMMA = {"instruction_part": "<start_of_turn>user\n", "response_part": "<start_of_turn>model\n",
          "image_size": None, "image_tokens": 256, "max_new_tokens": 256}
_QWEN = {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n",
         "image_size": 448, "image_tokens": 256, "max_new_tokens": 352}

MODELS = {spec.key: spec for spec in [
    ModelSpec("medgemma", "google/medgemma-1.5-4b-it", **_GEMMA,
              description="MedGemma 1.5 4B (Gemma 3): medical generalist, multilingual"),
    ModelSpec("chexone", "StanfordAIMI/CheXOne", **_QWEN,
              description="CheXOne 3B (Qwen2.5-VL-3B + CXR post-training): CXR specialist, English-only"),
    ModelSpec("qwen2.5-vl-3b", "Qwen/Qwen2.5-VL-3B-Instruct", **_QWEN,
              description="Qwen2.5-VL 3B Instruct: CheXOne's general-domain base (control)"),
]}
DEFAULT_MODEL = "medgemma"


def _match(name: str) -> ModelSpec | None:
    """Registry key, HF id, or a path / Unsloth mirror whose name contains the HF repo name."""
    name = name.lower().replace("\\", "/")
    for spec in MODELS.values():
        if name in (spec.key, spec.hf_id.lower()) or spec.hf_id.split("/")[-1].lower() in name:
            return spec
    return None


def get_spec(name: str) -> ModelSpec:
    spec = _match(name)
    if spec is None:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    return spec


def spec_for_adapter(path: str) -> ModelSpec:
    """Base model of an adapter dir (final_adapter / checkpoint-N) or of a plain HF id (zero-shot).
    Looks at run_args.json ("model", then "base_model"), then adapter_config.json, then the
    name itself. Runs trained before the registry existed have only "base_model" = MedGemma."""
    p = Path(path)
    candidates = []
    if (p / "run_args.json").exists():
        run_args = json.loads((p / "run_args.json").read_text())
        candidates += [run_args.get("model"), run_args.get("base_model")]
    if (p / "adapter_config.json").exists():
        candidates.append(json.loads((p / "adapter_config.json").read_text()).get("base_model_name_or_path"))
    candidates.append(str(path))
    for c in candidates:
        if c and (spec := _match(c)):
            return spec
    print(f"[models] can't tell the base model of {path}; assuming {DEFAULT_MODEL}")
    return MODELS[DEFAULT_MODEL]
