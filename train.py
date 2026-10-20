"""
QLoRA fine-tuning of a vision-language model for Indonesian CXR report generation, using
Unsloth. --model picks the base model (rrg/models.py): medgemma (default), chexone,
qwen2.5-vl-3b. Data, prompts, LoRA and optimizer settings are identical across models,
so runs with the same flags compare like for like.

One adapter covers both sections: each image yields a "findings" sample and/or an
"impression" sample, told apart by the prompt. Checkpoints are selected on eval_loss over
a fixed dev_internal subset (eval_loss is NOT comparable across models: different tokenizers).

    python train.py --run_name medgemma_full --freeze_vision --mimic_repeat 4 --lora_r 32 --lora_alpha 32
    python train.py --model chexone --run_name chexone_full --freeze_vision --mimic_repeat 4 --lora_r 32 --lora_alpha 32
    python train.py --model chexone --run_name chexone_smoke --max_train_samples 64 --max_steps 5 --eval_samples 16 --eval_steps 5
    python train.py --model chexone --run_name x --mimic_repeat 4 --dry_run    # sample counts only, no GPU
"""

import argparse
import gc
import inspect
import json

import pandas as pd

from rrg.config import CHECKPOINT_DIR, HINTS_PATH, MANIFEST_PATH, TASKS
from rrg.data import ReportSFTDataset, train_dev_samples
from rrg.models import DEFAULT_MODEL, MODELS, ModelSpec
from rrg.prompts import load_hints


def make_collator(model, processor, spec: ModelSpec):
    from unsloth.trainer import UnslothVisionDataCollator
    # unsloth's UnslothVisionDataCollator is a *args/**kwargs wrapper, so look through the MRO for the real signature
    supported = any("train_on_responses_only" in inspect.signature(cls.__init__).parameters
                    for cls in UnslothVisionDataCollator.__mro__ if "__init__" in vars(cls))
    if not supported:
        raise RuntimeError("this unsloth version can't mask prompts (train_on_responses_only); "
                           "the loss would include prompt and image tokens")
    return UnslothVisionDataCollator(model, processor, train_on_responses_only=True,
                                     instruction_part=spec.instruction_part, response_part=spec.response_part)


def check_loss_mask(collator, dataset, processor):
    """Print which tokens of the first sample the loss covers; fail if the prompt isn't masked."""
    labels = collator([dataset[0]])["labels"][0]
    kept = labels[labels != -100]
    text = processor.tokenizer.decode(kept)
    print(f"[train] loss covers {len(kept)}/{len(labels)} tokens of sample 0: {text[:160]!r}"
          f"{'...' if len(text) > 160 else ''}")
    if len(kept) == 0 or len(kept) > 0.8 * len(labels):
        raise RuntimeError("prompt masking looks wrong; check instruction_part/response_part in rrg/models.py "
                           "(python tools/check_model.py --model <key>)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True, help="checkpoints go to checkpoints/<run_name>/")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS), help="base model, see rrg/models.py")
    p.add_argument("--base_model", default=None, help="override the model's HF id / local path (same family)")
    p.add_argument("--dry_run", action="store_true", help="build the samples, print counts, exit (no GPU)")
    # data
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    p.add_argument("--frontal_only", action="store_true", help="drop lateral images from training")
    p.add_argument("--max_train_samples", type=int, default=None, help="random subset, for quick runs")
    p.add_argument("--eval_samples", type=int, default=256, help="fixed dev_internal subset for eval_loss")
    p.add_argument("--hints", action="store_true", help=f"add classifier hints from {HINTS_PATH.name} to prompts")
    p.add_argument("--hint_threshold", type=float, default=0.5)
    p.add_argument("--no_clean", action="store_true",
                   help="train on targets as in the manifest (skip rrg.text.clean_target)")
    p.add_argument("--mimic_repeat", type=int, default=1,
                   help="repeat MIMIC training samples this many times (MIMIC is ~2%% of the data)")
    # model / LoRA
    p.add_argument("--no_4bit", action="store_true", help="16-bit LoRA instead of QLoRA (needs much more VRAM)")
    p.add_argument("--freeze_vision", action="store_true", help="LoRA on the language model only")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    # optimization
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1, help="overrides --epochs when > 0")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=1,
                   help="eval runs while training memory is still allocated, so keep this small")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_length", type=int, default=1024, help="tokens incl. 256 image tokens")
    p.add_argument("--eval_steps", type=int, default=500, help="also the checkpoint interval")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in the run dir")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    spec = MODELS[args.model]
    args.base_model = args.base_model or spec.hf_id
    run_dir = CHECKPOINT_DIR / args.run_name

    manifest = pd.read_parquet(MANIFEST_PATH)
    hints = load_hints(args.hint_threshold) if args.hints else None
    train_samples, dev_samples, n_mimic = train_dev_samples(
        manifest, args.tasks, hints, clean=not args.no_clean, mimic_repeat=args.mimic_repeat,
        frontal_only=args.frontal_only, max_train_samples=args.max_train_samples,
        eval_samples=args.eval_samples, seed=args.seed)
    steps = -(-len(train_samples) // (args.batch_size * args.grad_accum))
    print(f"[train] model={spec.key} ({args.base_model})  image_size={spec.image_size or 'processor default'}")
    print(f"[train] train samples={len(train_samples):,} (mimic x{args.mimic_repeat}: {n_mimic:,} each)"
          f"  dev samples={len(dev_samples):,}  tasks={args.tasks}  ~{steps:,} optimizer steps/epoch")
    if args.dry_run:
        s = train_samples[0]
        print(f"[train] sample 0: {s['image_path']}\n--- prompt\n{s['prompt']}\n--- target\n{s['target']}")
        return

    import unsloth  # noqa: F401 - must be imported before transformers/trl/peft so its patches apply
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel, is_bf16_supported

    model, processor = FastVisionModel.from_pretrained(
        args.base_model, load_in_4bit=not args.no_4bit, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=not args.freeze_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        random_state=args.seed,
    )
    FastVisionModel.for_training(model)

    train_ds = ReportSFTDataset(train_samples, spec.image_size)
    collator = make_collator(model, processor, spec)
    check_loss_mask(collator, train_ds, processor)

    bf16 = is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=ReportSFTDataset(dev_samples, spec.image_size),
        args=SFTConfig(
            output_dir=str(run_dir),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            weight_decay=0.01,
            optim="adamw_8bit",
            bf16=bf16,
            fp16=not bf16,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            prediction_loss_only=True,  # don't keep (batch x seq x vocab) logits around during eval
            dataloader_num_workers=args.num_workers,
            seed=args.seed,
            report_to="none",
            # vision SFT: the collator does all tokenization
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_length,
        ),
    )

    # release cached activations/allocator blocks before each eval so it doesn't run out of VRAM
    original_evaluate = trainer.evaluate

    def evaluate_with_free_memory(*a, **kw):
        gc.collect()
        torch.cuda.empty_cache()
        result = original_evaluate(*a, **kw)
        torch.cuda.empty_cache()
        return result

    trainer.evaluate = evaluate_with_free_memory

    stats = trainer.train(resume_from_checkpoint=True if args.resume else None)
    print(f"[train] {stats}")

    final_dir = run_dir / "final_adapter"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    with open(final_dir / "run_args.json", "w") as f:
        json.dump({**vars(args), "best_checkpoint": trainer.state.best_model_checkpoint,
                   "best_eval_loss": trainer.state.best_metric}, f, indent=2)
    print(f"[train] best eval_loss={trainer.state.best_metric} -> adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
