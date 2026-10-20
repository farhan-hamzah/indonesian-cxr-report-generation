"""
ReportGenerator: one fine-tuned model (models/<key>/ adapter + its base model from Hugging Face),
loaded with plain transformers + peft (Unsloth is only needed for training).

Generation settings are the ones behind the reported test metrics (rrg_indo/infer.py):
greedy decoding, repetition_penalty 1.05, max_new_tokens 256 (MedGemma) / 352 (CheXOne),
one call per section (TEMUAN, KESAN), then post-processing.

    gen = ReportGenerator("medgemma")             # loads once (~1 min), reuse for every request
    gen.generate("cxr.png", view="PA")            # {"findings": "...", "impression": "..."}
    gen.generate_batch([img1, img2], views=["PA", "LATERAL"])

    chexone = ReportGenerator("chexone")          # a second model can sit next to the first one
"""

import torch

from .models import DEFAULT_MODEL, SKIP_4BIT_PATTERNS, get_spec
from .postprocess import postprocess
from .preprocess import load_image, model_input
from .prompts import TASKS, build_prompt, normalize_view


class ReportGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, base_model: str | None = None, device: str | None = None,
                 load_in_4bit: bool | None = None, max_new_tokens: int | None = None,
                 repetition_penalty: float = 1.05):
        """
        model          "medgemma" (default) or "chexone" (see cxr_report/models.py)
        base_model     local folder with the base model instead of the Hugging Face download
                       (MedGemma is gated: accept its terms on Hugging Face, then log in)
        device         "cuda", "cuda:1", "cpu"; default: cuda if available
        load_in_4bit   default: True on GPU (as in training, less VRAM), False on CPU
        max_new_tokens default per model: 256 MedGemma, 352 CheXOne (the same ~150 words)
        """
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.spec = get_spec(model)
        adapter_dir = self.spec.adapter_dir
        base_model = base_model or self.spec.base_model
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        on_gpu = device.startswith("cuda")
        if load_in_4bit is None:
            load_in_4bit = on_gpu
        if load_in_4bit and not on_gpu:
            raise ValueError("4-bit loading (bitsandbytes) needs an NVIDIA GPU; use load_in_4bit=False on CPU")
        # both base models overflow in float16, so GPUs without bfloat16 (e.g. T4, V100) run in float32
        self.dtype = torch.bfloat16 if (not on_gpu or torch.cuda.is_bf16_supported()) else torch.float32

        kwargs = {"dtype": self.dtype, "device_map": device}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype, llm_int8_skip_modules=SKIP_4BIT_PATTERNS)
        base = AutoModelForImageTextToText.from_pretrained(base_model, **kwargs)
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        # 16-bit: fold LoRA into the weights (faster); 4-bit keeps it separate, as in training
        self.model = (peft_model if load_in_4bit else peft_model.merge_and_unload()).eval()

        self.processor = AutoProcessor.from_pretrained(str(adapter_dir))
        self.processor.tokenizer.padding_side = "left"  # batched generation with a decoder-only model
        self.max_new_tokens = max_new_tokens or self.spec.max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.info = {"model": self.spec.key, "name": self.spec.name, "base_model": base_model,
                     "device": device, "load_in_4bit": load_in_4bit,
                     "dtype": str(self.dtype).replace("torch.", ""), "max_new_tokens": self.max_new_tokens}

    @torch.inference_mode()
    def _generate(self, images: list, prompts: list[str]) -> list[str]:
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        images = [model_input(im, self.spec.image_size) for im in images]
        inputs = self.processor(images=[[im] for im in images], text=texts, add_special_tokens=False,
                                padding=True, return_tensors="pt").to(self.model.device, dtype=self.dtype)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                                  repetition_penalty=self.repetition_penalty, use_cache=True,
                                  pad_token_id=self.processor.tokenizer.pad_token_id)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(new_tokens, skip_special_tokens=True)]

    def generate_batch(self, images: list, views: list | None = None, tasks=TASKS,
                       return_raw: bool = False) -> list[dict]:
        """
        images  paths, bytes, file objects, numpy arrays or PIL images (see preprocess.load_image)
        views   per image "AP", "PA", "LATERAL" (or "LL"/"RL"), or None if unknown
        tasks   "findings" (TEMUAN) and/or "impression" (KESAN)
        returns one dict per image: {"findings": ..., "impression": ...}, plus "<task>_raw"
                (output before post-processing) if return_raw
        Batches above ~8 images may run out of memory on a 16 GB GPU; split them.
        """
        views = list(views) if views is not None else [None] * len(images)
        if len(views) != len(images):
            raise ValueError(f"{len(images)} images but {len(views)} views")
        unknown = set(tasks) - set(TASKS)
        if unknown:
            raise ValueError(f"unknown task(s) {sorted(unknown)}; use {TASKS}")
        views = [normalize_view(v) for v in views]
        pil = [load_image(im) for im in images]
        results = [{} for _ in pil]
        for task in tasks:
            texts = self._generate(pil, [build_prompt(task, v) for v in views])
            for res, text in zip(results, texts):
                res[task] = postprocess(text)
                if return_raw:
                    res[f"{task}_raw"] = text
        return results

    def generate(self, image, view: str | None = None, tasks=TASKS, return_raw: bool = False) -> dict:
        """One image -> {"findings": TEMUAN text, "impression": KESAN text}."""
        return self.generate_batch([image], [view], tasks, return_raw)[0]
