#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import importlib.metadata
import os
import sys
import time

os.environ.setdefault("PT_HPU_LAZY_MODE", "0")

import torch
import torch.distributed as dist
import habana_frameworks.torch.core as htcore
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


MODEL_ID = "Qwen/Qwen3.6-27B-FP8"
FP8_CORRECTNESS_FALLBACKS = {
    "Qwen/Qwen3.6-27B-FP8": "Qwen/Qwen3.6-27B",
}
QWEN3_TEXT_TP_PLAN = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
QWEN3_5_TP_PLAN = {
    **{f"model.language_model.{key}": value for key, value in QWEN3_TEXT_TP_PLAN.items()},
    "lm_head": "colwise_gather_output",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen/Qwen3.6-27B-FP8 on Gaudi HPU.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="日本語で短く自己紹介して")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--benchmark-runs", type=int, default=1)
    parser.add_argument("--hide-output", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--disable-fp8-correctness-fallback", action="store_true")
    return parser.parse_args()


def print_versions() -> None:
    for package in ("torch", "habana-torch-plugin", "optimum-habana", "transformers"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: not installed")
    print(f"HPU available: {torch.hpu.is_available()}, devices: {torch.hpu.device_count()}")


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def print_main(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def setup_tensor_parallel(tp_size: int) -> None:
    if tp_size < 1:
        raise ValueError("--tensor-parallel-size must be >= 1.")
    if tp_size == 1:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != tp_size:
        raise RuntimeError(
            "Tensor parallel requires torchrun with matching process count. "
            f"Run with: torchrun --standalone --nproc_per_node={tp_size} "
            "run_qwen36_hpu.py --tensor-parallel-size "
            f"{tp_size} ..."
        )
    if torch.hpu.device_count() < tp_size:
        raise RuntimeError(
            f"--tensor-parallel-size={tp_size} requested, but only "
            f"{torch.hpu.device_count()} HPU devices are available."
        )


def get_input_device(tp_size: int) -> str:
    return "hpu"


def get_tensor_parallel_plan(model_id: str) -> str | dict[str, str]:
    if model_id.startswith("Qwen/Qwen3.6"):
        return QWEN3_5_TP_PLAN
    return "auto"


def is_qwen36_model(model_id: str) -> bool:
    return model_id.startswith("Qwen/Qwen3.6")


def is_fp8_model(model_id: str) -> bool:
    return model_id.upper().endswith("-FP8")


def resolve_execution_model_id(model_id: str, disable_fp8_fallback: bool) -> str:
    if disable_fp8_fallback:
        return model_id
    if torch.hpu.is_available() and model_id in FP8_CORRECTNESS_FALLBACKS:
        fallback_model_id = FP8_CORRECTNESS_FALLBACKS[model_id]
        print_main(
            f"{model_id} produces non-finite logits on the current HPU Transformers FP8 path; "
            f"using {fallback_model_id} for correctness."
        )
        return fallback_model_id
    return model_id


def enable_optimum_habana() -> bool:
    python_bin_dir = os.path.dirname(sys.executable)
    os.environ["PATH"] = f"{python_bin_dir}:{os.environ.get('PATH', '')}"
    try:
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    except Exception as error:
        print_main(f"Optimum Habana unavailable: {error}")
        return False

    adapt_transformers_to_gaudi()
    print_main("Optimum Habana Gaudi patches enabled.")
    return True


class GenerationTimer(StoppingCriteria):
    def __init__(self, prompt_len: int) -> None:
        self.prompt_len = prompt_len
        self.start_time: float | None = None
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_tokens = input_ids.shape[-1] - self.prompt_len
        if generated_tokens > 0:
            now = time.perf_counter()
            if self.first_token_time is None:
                self.first_token_time = now
            self.last_token_time = now
        return False

    @property
    def ttft(self) -> float | None:
        if self.start_time is None or self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time


@dataclass
class GenerationMetrics:
    run_label: str
    generated_tokens: int
    ttft: float | None
    eerl: float
    tps: float
    e2e_tps: float
    text: str


def get_eos_token_id(tokenizer: AutoTokenizer | AutoProcessor) -> int:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return eos_token_id
    return tokenizer.tokenizer.eos_token_id


def generate_once(
    model: AutoModelForCausalLM | AutoModelForImageTextToText,
    tokenizer: AutoTokenizer | AutoProcessor,
    inputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
    run_label: str,
) -> GenerationMetrics:
    prompt_len = inputs["input_ids"].shape[-1]
    timer = GenerationTimer(prompt_len)
    do_sample = args.temperature > 0

    torch.hpu.synchronize()
    timer.start()
    with torch.inference_mode():
        generation_kwargs = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": get_eos_token_id(tokenizer),
            "use_cache": True,
            "stopping_criteria": StoppingCriteriaList([timer]),
        }
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        output_ids = model.generate(
            **generation_kwargs,
        )
        htcore.mark_step()
        torch.hpu.synchronize()
    eerl = time.perf_counter() - (timer.start_time or time.perf_counter())

    generated_ids = output_ids[:, prompt_len:]
    generated_tokens = generated_ids.shape[-1]
    ttft = timer.ttft
    decode_time = max(eerl - (ttft or 0.0), 1e-9)
    tps_tokens = max(generated_tokens - 1, 0) if ttft is not None else generated_tokens
    tps = tps_tokens / decode_time
    e2e_tps = generated_tokens / max(eerl, 1e-9)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return GenerationMetrics(
        run_label=run_label,
        generated_tokens=generated_tokens,
        ttft=ttft,
        eerl=eerl,
        tps=tps,
        e2e_tps=e2e_tps,
        text=text,
    )


def print_metrics(metrics: GenerationMetrics) -> None:
    ttft = f"{metrics.ttft:.3f}s" if metrics.ttft is not None else "n/a"
    print(
        f"{metrics.run_label}: "
        f"tokens={metrics.generated_tokens}, "
        f"TTFT={ttft}, "
        f"EERL={metrics.eerl:.3f}s, "
        f"TPS={metrics.tps:.2f}, "
        f"e2e_TPS={metrics.e2e_tps:.2f}"
    )


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0 or args.benchmark_runs < 1:
        raise ValueError("--warmup-runs must be >= 0 and --benchmark-runs must be >= 1.")
    setup_tensor_parallel(args.tensor_parallel_size)

    if is_main_process():
        print_versions()
    if not torch.hpu.is_available():
        raise RuntimeError("HPU is not available. Check Habana driver/runtime setup.")

    htcore.hpu_inference_set_env()
    optimum_enabled = enable_optimum_habana()
    execution_model_id = resolve_execution_model_id(
        args.model_id,
        args.disable_fp8_correctness_fallback,
    )
    if is_qwen36_model(execution_model_id):
        tokenizer = AutoProcessor.from_pretrained(execution_model_id)
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        inputs = tokenizer(text=[text], return_tensors="pt")
        model_cls = AutoModelForImageTextToText
        model_kwargs = {} if is_fp8_model(execution_model_id) else {"dtype": torch.bfloat16}
    else:
        tokenizer = AutoTokenizer.from_pretrained(execution_model_id)
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt")
        model_cls = AutoModelForCausalLM
        model_kwargs = {} if is_fp8_model(execution_model_id) else {"torch_dtype": torch.bfloat16}

    start = time.time()
    load_kwargs = {
        **model_kwargs,
        "low_cpu_mem_usage": True,
    }
    if args.tensor_parallel_size > 1:
        load_kwargs["tp_plan"] = get_tensor_parallel_plan(execution_model_id)
        load_kwargs["tp_size"] = args.tensor_parallel_size
    else:
        load_kwargs["device_map"] = {"": "hpu"}

    model = model_cls.from_pretrained(
        execution_model_id,
        **load_kwargs,
    )
    model.eval()
    for key in ("input_ids", "attention_mask"):
        if key in inputs:
            inputs[key] = inputs[key].to(dtype=torch.int32)
    inputs = {key: value.to(get_input_device(args.tensor_parallel_size)) for key, value in inputs.items()}
    htcore.mark_step()
    precision = "fp8 pretrained" if is_fp8_model(execution_model_id) else "bf16"
    if execution_model_id != args.model_id:
        precision = f"{precision}, fallback from {args.model_id}"
    print_main(f"Loaded model in {time.time() - start:.1f}s ({precision})")
    if args.tensor_parallel_size > 1:
        print_main(f"Tensor parallel enabled with {args.tensor_parallel_size} HPU processes.")
    if not optimum_enabled:
        print_main("Continuing with Transformers + habana_frameworks fallback.")

    for index in range(args.warmup_runs):
        metrics = generate_once(model, tokenizer, inputs, args, f"warmup {index + 1}")
        if is_main_process():
            print_metrics(metrics)

    benchmark_metrics = []
    for index in range(args.benchmark_runs):
        metrics = generate_once(model, tokenizer, inputs, args, f"run {index + 1}")
        benchmark_metrics.append(metrics)
        if is_main_process():
            print_metrics(metrics)
            if not args.hide_output:
                print(metrics.text)

    if is_main_process() and len(benchmark_metrics) > 1:
        valid_ttfts = [metric.ttft for metric in benchmark_metrics if metric.ttft is not None]
        avg_ttft = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else None
        avg_metrics = GenerationMetrics(
            run_label="avg",
            generated_tokens=round(
                sum(metric.generated_tokens for metric in benchmark_metrics) / len(benchmark_metrics)
            ),
            ttft=avg_ttft,
            eerl=sum(metric.eerl for metric in benchmark_metrics) / len(benchmark_metrics),
            tps=sum(metric.tps for metric in benchmark_metrics) / len(benchmark_metrics),
            e2e_tps=sum(metric.e2e_tps for metric in benchmark_metrics) / len(benchmark_metrics),
            text="",
        )
        print_metrics(avg_metrics)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
