#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import importlib.metadata
import os
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HF_HOME = PROJECT_ROOT / "hf_cache"
DEFAULT_MODEL_ID = "Qwen/Qwen3-235B-A22B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CLI chat bot with vLLM Gaudi.")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--prompt", default="日本語で短く自己紹介して")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("VLLM_TP_SIZE", "8")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(DEFAULT_HF_HOME)))
    parser.add_argument("--revision", default=os.environ.get("HF_MODEL_REVISION", "main"))
    parser.add_argument("--dtype", default=os.environ.get("VLLM_DTYPE", "bfloat16"))
    parser.add_argument("--once", action="store_true", help="Run --prompt once and exit instead of starting chat.")
    parser.add_argument("--max-history-turns", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", help="Keep vLLM INFO/WARNING logs visible.")
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def hf_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return DEFAULT_HF_HOME / "hub"


def hf_model_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def local_snapshot_path(model_id: str, revision: str) -> Path | None:
    model_dir = hf_cache_root() / hf_model_cache_name(model_id)
    ref_path = model_dir / "refs" / revision
    if ref_path.exists():
        snapshot_revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = model_dir / "snapshots" / snapshot_revision
        if snapshot_revision and snapshot_path.exists():
            return snapshot_path

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


def resolve_model_source(model_id: str, revision: str, local_files_only: bool) -> str:
    if not local_files_only:
        return model_id

    snapshot_path = local_snapshot_path(model_id, revision)
    if snapshot_path is None:
        raise RuntimeError(
            f"Local snapshot for {model_id} was not found in {hf_cache_root()}. "
            f"Run: {os.environ.get('PYTHON_BIN', 'python')} download_hf_models.py --model-id {model_id} --prepare"
        )
    return str(snapshot_path)


def render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@dataclass
class TurnMetrics:
    text: str
    generated_tokens: int
    prompt_tokens: int
    ttft_sec: float | None
    elapsed_sec: float
    tps: float
    e2e_tps: float


def sampling_params_for(args: argparse.Namespace):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        output_kind=RequestOutputKind.DELTA,
    )


def load_llm(model_source: str, args: argparse.Namespace):
    from vllm import LLM

    llm_kwargs = {
        "model": model_source,
        "tokenizer": model_source,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
    }
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    auto_expert_parallel = any(marker in args.model_id for marker in ("A3B", "A22B", "DeepSeek-V2"))
    if args.enable_expert_parallel or auto_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    return LLM(**llm_kwargs)


def generate_turn(llm, prompt: str, args: argparse.Namespace, turn_index: int, stream: bool = True) -> TurnMetrics:
    from vllm.outputs import RequestOutput

    sampling_params = sampling_params_for(args)
    engine_input = llm._preprocess_cmpl_one(prompt, tokenization_kwargs=None, mm_processor_kwargs=None)
    request_id = f"chat-{turn_index}-{uuid.uuid4().hex}"

    started = time.perf_counter()
    llm.llm_engine.add_request(request_id, engine_input, sampling_params, priority=0)

    chunks: list[str] = []
    generated_tokens = 0
    prompt_tokens = 0
    first_token_time: float | None = None

    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if not isinstance(output, RequestOutput) or output.request_id != request_id:
                continue
            if output.prompt_token_ids is not None:
                prompt_tokens = len(output.prompt_token_ids)
            for completion in output.outputs:
                token_count = len(completion.token_ids)
                if token_count > 0 and first_token_time is None:
                    first_token_time = time.perf_counter()
                generated_tokens += token_count
                if completion.text:
                    chunks.append(completion.text)
                    if stream:
                        print(completion.text, end="", flush=True)

    finished = time.perf_counter()
    text = "".join(chunks).strip()
    elapsed_sec = finished - started
    ttft_sec = None if first_token_time is None else first_token_time - started
    decode_sec = max(elapsed_sec - (ttft_sec or 0.0), 1e-9)
    tps_tokens = max(generated_tokens - 1, 0) if ttft_sec is not None else generated_tokens
    tps = tps_tokens / decode_sec
    e2e_tps = generated_tokens / max(elapsed_sec, 1e-9)

    if stream:
        print()

    return TurnMetrics(
        text=text,
        generated_tokens=generated_tokens,
        prompt_tokens=prompt_tokens,
        ttft_sec=ttft_sec,
        elapsed_sec=elapsed_sec,
        tps=tps,
        e2e_tps=e2e_tps,
    )


def trim_history(messages: list[dict[str, str]], max_history_turns: int) -> list[dict[str, str]]:
    system_messages = [message for message in messages if message["role"] == "system"]
    chat_messages = [message for message in messages if message["role"] != "system"]
    keep_messages = chat_messages[-max(0, max_history_turns) * 2 :] if max_history_turns > 0 else []
    return system_messages + keep_messages


def print_metrics(metrics: TurnMetrics) -> None:
    ttft = f"{metrics.ttft_sec:.3f}s" if metrics.ttft_sec is not None else "n/a"
    print(
        "[metrics] "
        f"prompt_tokens={metrics.prompt_tokens} "
        f"generated_tokens={metrics.generated_tokens} "
        f"TTFT={ttft} "
        f"elapsed={metrics.elapsed_sec:.3f}s "
        f"TPS={metrics.tps:.2f} "
        f"e2e_TPS={metrics.e2e_tps:.2f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if not args.verbose:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.pop("VLLM_TP_SIZE", None)
    os.environ.pop("VLLM_MAX_MODEL_LEN", None)
    if args.skip_warmup:
        os.environ["VLLM_SKIP_WARMUP"] = "true"

    print(f"torch={package_version('torch')}")
    print(f"habana-torch-plugin={package_version('habana-torch-plugin')}")
    print(f"vllm={package_version('vllm')}")
    print(f"vllm-gaudi={package_version('vllm-gaudi')}")
    print(f"model_id={args.model_id}")
    print(f"tensor_parallel_size={args.tensor_parallel_size}")

    from vllm import LLM, SamplingParams

    model_source = resolve_model_source(args.model_id, args.revision, args.local_files_only)
    print(f"model_source={model_source}")

    started = time.perf_counter()
    llm = load_llm(model_source, args)
    load_sec = time.perf_counter() - started
    print(f"loaded_sec={load_sec:.2f}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    if args.once:
        messages.append({"role": "user", "content": args.prompt})
        prompt = render_chat_prompt(tokenizer, messages)
        print("\nassistant> ", end="", flush=True)
        metrics = generate_turn(llm, prompt, args, turn_index=1, stream=True)
        print_metrics(metrics)
        return

    print("\nCLI chat is ready. Commands: /exit, /quit, /reset, /help")
    turn_index = 0
    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("bye")
            break
        if user_text == "/help":
            print("commands: /exit, /quit, /reset, /help")
            continue
        if user_text == "/reset":
            messages = [{"role": "system", "content": args.system_prompt}] if args.system_prompt else []
            print("history reset")
            continue

        turn_index += 1
        messages.append({"role": "user", "content": user_text})
        messages = trim_history(messages, args.max_history_turns)
        prompt = render_chat_prompt(tokenizer, messages)
        print("assistant> ", end="", flush=True)
        metrics = generate_turn(llm, prompt, args, turn_index=turn_index, stream=True)
        messages.append({"role": "assistant", "content": metrics.text})
        print_metrics(metrics)


if __name__ == "__main__":
    main()
