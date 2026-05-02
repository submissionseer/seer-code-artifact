#!/usr/bin/env python3
"""Canonical DPO/IPO training entrypoint (TRL + QLoRA)."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

ALPACA_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

INSTRUCTION = (
    "Generate a search query to find information needed to answer "
    "the following question. Use the provided context if available."
)


def parse_prompt_fields(prompt_str: str) -> tuple[str, str]:
    """Parse instruction/input from stored prompt format."""
    instruction = INSTRUCTION
    input_text = prompt_str
    if prompt_str.startswith("Instruction: "):
        parts = prompt_str.split("\nInput: ", 1)
        if len(parts) == 2:
            instruction = parts[0].replace("Instruction: ", "", 1).strip()
            input_text = parts[1].strip()
    return instruction, input_text


def load_dpo_dataset(path: str) -> Dataset:
    """Load JSONL preference pairs into TRL format."""
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            instruction, input_text = parse_prompt_fields(row["prompt"])
            prompt = ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)
            records.append(
                {
                    "prompt": prompt,
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
            )
    return Dataset.from_list(records)


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def maybe_resolve_config_relative_path(path_value: str | None, config_path: str | None) -> str | None:
    """Resolve relative path values against the config file directory."""
    if not path_value:
        return path_value
    p = Path(path_value)
    if p.is_absolute() or not config_path:
        return str(p)
    return str((Path(config_path).resolve().parent / p).resolve())


def filter_kwargs_for_signature(fn: object, kwargs: dict) -> dict:
    sig = inspect.signature(fn)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid}


def get_report_to(value: str) -> str:
    """Return safe report_to value even if wandb package is missing."""
    if value != "wandb":
        return value
    try:
        import wandb  # noqa: F401
    except Exception:
        print("wandb package not available; falling back to report_to='none'")
        return "none"
    return "wandb"


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO/IPO training with TRL + QLoRA")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print settings only")

    parser.add_argument("--base-model", default=None)
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--loss-type", default=None, choices=["sigmoid", "ipo", "hinge"])
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)

    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)

    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)

    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--report-to", default=None, choices=["wandb", "none"])
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def merged_config(cli_args: argparse.Namespace) -> argparse.Namespace:
    def _get_config_value(config_obj: dict, key: str):
        if key in config_obj:
            return config_obj[key]
        dashed = key.replace("_", "-")
        if dashed in config_obj:
            return config_obj[dashed]
        return None

    defaults = {
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "val_split": 0.05,
        "loss_type": "ipo",
        "beta": 0.1,
        "learning_rate": 5e-7,
        "num_epochs": 2,
        "warmup_steps": 150,
        "max_length": 2048,
        "max_prompt_length": 1792,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "bf16": True,
        "wandb_project": "seer-experiment",
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 250,
        "seed": 42,
        "report_to": "wandb",
    }

    config = {}
    if cli_args.config:
        config = load_config(cli_args.config)
        print(f"Loaded config from {cli_args.config}")

    final: dict[str, object] = {}
    for key in defaults:
        cli_val = getattr(cli_args, key, None)
        config_val = _get_config_value(config, key)
        if cli_val is not None:
            final[key] = cli_val
        elif config_val is not None:
            final[key] = config_val
        else:
            final[key] = defaults[key]

    for key in ["sft_adapter", "train_data", "output_dir"]:
        cli_val = getattr(cli_args, key, None)
        config_val = _get_config_value(config, key)
        if cli_val is not None:
            val = cli_val
        elif config_val is not None:
            val = config_val
        else:
            raise ValueError(f"--{key.replace('_', '-')} is required (via CLI or config)")
        final[key] = maybe_resolve_config_relative_path(val, cli_args.config)

    cli_val = getattr(cli_args, "wandb_run_name", None)
    config_val = _get_config_value(config, "wandb_run_name")
    final["wandb_run_name"] = cli_val or config_val

    type_map = {
        "beta": float,
        "learning_rate": float,
        "val_split": float,
        "lora_dropout": float,
        "num_epochs": int,
        "warmup_steps": int,
        "max_length": int,
        "max_prompt_length": int,
        "lora_r": int,
        "lora_alpha": int,
        "micro_batch_size": int,
        "gradient_accumulation_steps": int,
        "logging_steps": int,
        "save_steps": int,
        "eval_steps": int,
        "seed": int,
    }
    for key, typ in type_map.items():
        if key in final and final[key] is not None:
            final[key] = typ(final[key])

    final["report_to"] = get_report_to(str(final.get("report_to", "wandb")))
    return argparse.Namespace(**final)


def build_dpo_config(args: argparse.Namespace) -> DPOConfig:
    from trl import DPOConfig

    run_name = args.wandb_run_name or f"dpo-{args.loss_type}-b{args.beta}-lr{args.learning_rate}"

    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.micro_batch_size,
        "per_device_eval_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "seed": args.seed,
        "report_to": args.report_to,
        "run_name": run_name,
        "gradient_checkpointing": args.gradient_checkpointing,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "precompute_ref_log_probs": True,
        "remove_unused_columns": False,
    }

    # transformers/trl arg names vary by version
    dpo_params = inspect.signature(DPOConfig.__init__).parameters
    if "eval_strategy" in dpo_params:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = args.eval_steps
    elif "evaluation_strategy" in dpo_params:
        kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = args.eval_steps

    kwargs = filter_kwargs_for_signature(DPOConfig.__init__, kwargs)
    return DPOConfig(**kwargs)


def build_trainer(
    model,
    training_args: DPOConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
) -> DPOTrainer:
    from trl import DPOTrainer

    kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
    }
    params = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" not in params and "tokenizer" in params:
        kwargs.pop("processing_class", None)
        kwargs["tokenizer"] = tokenizer
    kwargs = filter_kwargs_for_signature(DPOTrainer.__init__, kwargs)
    return DPOTrainer(**kwargs)


def find_latest_checkpoint(output_dir: str) -> str | None:
    root = Path(output_dir)
    if not root.exists():
        return None

    checkpoints: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        # Trainer resume needs trainer_state plus optimizer/scheduler state.
        required = [
            path / "trainer_state.json",
            path / "optimizer.pt",
            path / "scheduler.pt",
            path / "rng_state.pth",
        ]
        if not all(p.exists() for p in required):
            continue
        checkpoints.append((step, path))

    if not checkpoints:
        return None

    checkpoints.sort()
    return str(checkpoints[-1][1])


def main() -> None:
    cli_args = build_args()
    args = merged_config(cli_args)

    print("=" * 60)
    print("DPO/IPO Training")
    print("=" * 60)
    print(f"  Base model: {args.base_model}")
    print(f"  SFT adapter: {args.sft_adapter}")
    print(f"  Train data: {args.train_data}")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Beta: {args.beta}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    if cli_args.dry_run:
        print("\nDry run only; skipping dataset/model loading.")
        return

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("\nLoading DPO dataset...")
    dataset = load_dpo_dataset(args.train_data)
    print(f"  Total pairs: {len(dataset)}")
    split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading base model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading SFT adapter from {args.sft_adapter}...")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    print("  SFT adapter merged into base model")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = build_dpo_config(args)

    print("\nInitializing DPOTrainer...")
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    latest_checkpoint = find_latest_checkpoint(args.output_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    print(f"\nSaving final model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("DPO training complete!")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
