"""
Shared library for the Indonesian CXR report-generation pipeline.

  config       paths and split roles
  models       base VLMs that can be fine-tuned (MedGemma, CheXOne, Qwen2.5-VL) and what differs between them
  prompts      task prompts and optional classifier hints
  text         text normalization for references and training targets
  data         manifest -> training samples, image loading
  postprocess  repetition cleanup for generated reports
  metrics      Indonesian NLG metrics

Modules here import nothing heavier than torch/pandas at import time, so scripts can
import Unsloth first (it must come before transformers/trl/peft).
"""
