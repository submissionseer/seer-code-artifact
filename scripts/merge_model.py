#!/usr/bin/env python3
"""
Merge LoRA adapters (SFT + IPO) into base model weights for iteration.

Usage:
    python3 scripts/merge_model.py \
        --base-model meta-llama/Meta-Llama-3-8B-Instruct \
        --sft-adapter /workspace/data/seer_sft/qlora-out-jina-mmr-5k-k3 \
        --ipo-adapter /workspace/data/dpo_out/dpo-mmr-5k-ipo-r64 \
        --output /workspace/data/merged_models/mmr-iter1-merged
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--sft-adapter", required=True, help="SFT LoRA adapter path")
    parser.add_argument("--ipo-adapter", default=None, help="IPO LoRA adapter path (optional)")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Merge SFT adapter
    print(f"Loading and merging SFT adapter: {args.sft_adapter}")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    print("  SFT adapter merged")

    # Merge IPO adapter if provided
    if args.ipo_adapter:
        print(f"Loading and merging IPO adapter: {args.ipo_adapter}")
        model = PeftModel.from_pretrained(model, args.ipo_adapter)
        model = model.merge_and_unload()
        print("  IPO adapter merged")

    # Save
    print(f"Saving merged model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
