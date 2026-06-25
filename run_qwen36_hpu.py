#!/usr/bin/env python3
import argparse
import importlib.metadata
import os
import time

import torch
import habana_frameworks.torch.core as htcore
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "Qwen/Qwen3.6-27B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen/Qwen3.6-27B on Gaudi HPU.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="日本語で短く自己紹介して")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def print_versions() -> None:
    for package in ("torch", "habana-torch-plugin", "optimum-habana", "transformers"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: not installed")
    print(f"HPU available: {torch.hpu.is_available()}, devices: {torch.hpu.device_count()}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PT_HPU_LAZY_MODE", "0")

    print_versions()
    if not torch.hpu.is_available():
        raise RuntimeError("HPU is not available. Check Habana driver/runtime setup.")

    processor = AutoProcessor.from_pretrained(args.model_id)
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = processor(text=[text], return_tensors="pt")

    start = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "hpu"},
    )
    model.eval()
    inputs = {key: value.to("hpu") for key, value in inputs.items()}
    htcore.mark_step()
    print(f"Loaded model in {time.time() - start:.1f}s")

    do_sample = args.temperature > 0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        htcore.mark_step()
        torch.hpu.synchronize()

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, prompt_len:]
    print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip())


if __name__ == "__main__":
    main()
