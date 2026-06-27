# Gaudi demo

Run Qwen models on Intel Gaudi HPU. The chat server defaults to
`Qwen/Qwen3.6-27B`.

## Docs

- [AI Agent Specification with Intel Gaudi](docs/ai_agent_gaudi_spec.md)

## Chat UI

Start a web chat server bound to all interfaces. The server uses port `8000` by
default:

```bash
HF_HOME=$PWD/hf_cache /home/test1/habanalabs-venv/bin/python chat_server.py \
  --host 0.0.0.0
```

Then open `http://<server-ip>:8000/`.

You can override the default with `SERVER_PORT` or `--port`.

To keep the server running in the background after closing the terminal, start it
with `nohup`:

```bash
cd /home/test1/kazuki/gaudi_demo

HF_HOME=$PWD/hf_cache nohup /home/test1/habanalabs-venv/bin/python chat_server.py \
  --host 0.0.0.0 \
  > chat_server.log 2>&1 &
```

Check the startup log:

```bash
tail -f chat_server.log
```

Stop the background server:

```bash
pkill -f "chat_server.py --host 0.0.0.0"
```

The first screen asks for a user name. Each user gets a separate chat screen and
history, saved in `chat_history.json`.

The UI can switch between:

- `Qwen/Qwen3.6-27B`
- `Qwen/Qwen3.6-27B-FP8`
- `Qwen/Qwen3.6-35B-A3B-FP8`
- `Qwen/Qwen3-32B`
- `Qwen/Qwen3-235B-A22B`

The reasoning strength selector changes the speed/depth tradeoff:

- `Low`: shortest responses, fastest, thinking disabled
- `Medium`: standard responses, thinking disabled
- `High`: longer responses, thinking enabled

The agent mode selector is applied per message:

- `Auto`: decide whether web search is needed, then answer with or without sources
- `Chat`: answer with the selected model only
- `Deep search`: run multiple DuckDuckGo searches, then answer with broader source context

DuckDuckGo is the default search engine. To temporarily switch back to Bing,
start the server with `SEARCH_ENGINE=bing`.

The chat screen supports multiple threads per user. Use the left-side thread
list to switch topics without mixing conversation history. New thread titles are
generated automatically from the first user message. The default `Auto` mode
makes a per-message search decision inside the active thread.

While a message is running, the UI shows agent steps such as web search,
source preparation, prompt construction, model generation, and completion.
Messages are submitted as asynchronous jobs, so you can continue sending follow-up
prompts while previous generations are queued or running. HPU generation is still
serialized by the server-side model lock on a single HPU.

When you choose a different model, the server unloads the current model and loads
the selected one on the next chat request.

`Qwen/Qwen3.6-35B-A3B-FP8` uses the pretrained FP8 weights from Hugging Face.
It remains available in the model selector alongside the default 27B bf16 model.

## Performance notes

The built-in FastAPI server uses Transformers directly on a single HPU. On an
8-card Gaudi2 host this leaves the other HPUs idle; use vLLM for Intel Gaudi or
DeepSpeed/Optimum Habana tensor parallel inference for production throughput.

`Qwen/Qwen3.6-35B-A3B-FP8` currently emits a Transformers warning that the FP8
checkpoint is dequantized to bf16 on this HPU path, so it should not be treated
as full FP8 compute. The server enables Habana inference settings, int32 token
inputs, and KV cache explicitly, but the main remaining bottleneck is the
single-HPU Transformers execution path.

```bash
HF_HOME=$PWD/hf_cache /home/test1/habanalabs-venv/bin/python chat_server.py \
  --host 0.0.0.0 \
  --model-id Qwen/Qwen3.6-27B
```

## CLI

```bash
HF_HOME=$PWD/hf_cache /home/test1/habanalabs-venv/bin/python run_qwen36_hpu.py \
  --prompt "日本語で短く自己紹介して"
```

The CLI defaults to `Qwen/Qwen3.6-27B-FP8`. To run the bf16 checkpoint instead,
pass `--model-id Qwen/Qwen3.6-27B`.

On the current Gaudi HPU + Transformers path, `Qwen/Qwen3.6-27B-FP8` produces
non-finite logits after the built-in FP8-to-bf16 fallback. The demo therefore
accepts the FP8 model ID but automatically executes the sibling bf16 checkpoint
`Qwen/Qwen3.6-27B` for correct text output. Pass
`--disable-fp8-correctness-fallback` only when debugging the raw FP8 path.

This machine has Habana software 1.24 installed. The working Python environment is
`/home/test1/habanalabs-venv`.

Note: `Qwen/Qwen3.6-27B` requires a newer Transformers release with `qwen3_5`
support. `optimum-habana==1.21.0` is installed in the environment, but its public
wrappers are currently pinned to `transformers<4.56`, so this demo uses the
Habana PyTorch bridge directly while keeping the Gaudi/Optimum Habana environment
installed.

## Optimum Habana compatibility venv

An alternate environment for Optimum Habana compatibility is available at:

```bash
/home/test1/habanalabs-venv-optimum
```

It keeps the Habana runtime stack and uses:

```text
optimum-habana==1.21.0
transformers==4.55.4
huggingface_hub==0.36.2
tokenizers==0.21.4
```

Because Optimum Habana checks the Habana package version by running `pip`, put
the venv `bin` directory first in `PATH`:

```bash
PATH=/home/test1/habanalabs-venv-optimum/bin:$PATH \
HF_HOME=$PWD/hf_cache \
/home/test1/habanalabs-venv-optimum/bin/python run_qwen36_hpu.py \
  --model-id Qwen/Qwen3-32B \
  --prompt "日本語で短く自己紹介して"
```

For tensor parallel:

```bash
PATH=/home/test1/habanalabs-venv-optimum/bin:$PATH \
HF_HOME=$PWD/hf_cache PT_HPU_WEIGHT_SHARING=0 \
/home/test1/habanalabs-venv-optimum/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  run_qwen36_hpu.py \
  --model-id Qwen/Qwen3-32B \
  --tensor-parallel-size 2 \
  --prompt "日本語で短く自己紹介して"
```

This venv supports Qwen3 models such as `Qwen/Qwen3-32B`. It does not support
Qwen3.6 checkpoints such as `Qwen/Qwen3.6-27B`, because those require the newer
`qwen3_5` architecture in Transformers 5.x.

`Qwen/Qwen3-235B-A22B` is available as a Qwen3 causal LM option. The current demo
uses a single-HPU Transformers placement, so this checkpoint requires enough HPU
memory for the full model or a future multi-HPU placement path before it can be
used reliably.

`Qwen/Qwen3-32B-FP8` is accepted by the CLI, but the current HPU + Optimum Habana
FP8 path creates an FP8 KV cache while the model produces bf16 key/value states.
The demo therefore falls back to `Qwen/Qwen3-32B` for correct generation.
