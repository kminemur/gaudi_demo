# Gaudi demo

Run Qwen models on Intel Gaudi HPU. The chat server defaults to
`Qwen/Qwen3-32B`.

## Docs

- [AI Agent Specification with Intel Gaudi](docs/ai_agent_gaudi_spec.md)

## Model download script

Use the dedicated script to pre-download and verify model snapshots before
starting the chat server. This keeps the UI from appearing stuck while large
model shards are still being fetched. If `HF_HOME` is not set, the script uses
this repository's `./hf_cache` directory by default.

Prepare all demo default models:

```bash
/home/test1/habanalabs-venv-optimum/bin/python download_hf_models.py \
  --all-defaults \
  --prepare
```

Prepare specific models only:

```bash
/home/test1/habanalabs-venv-optimum/bin/python download_hf_models.py \
  --model-id Qwen/Qwen3-32B \
  --model-id Qwen/Qwen3-235B-A22B \
  --prepare
```

If authentication is required, set a token:

```bash
HF_TOKEN=<your_token> /home/test1/habanalabs-venv-optimum/bin/python download_hf_models.py \
  --model-id Qwen/Qwen3-235B-A22B \
  --prepare
```

If a previous download was interrupted, `--prepare` removes stale
`*.incomplete` blob files for the selected models and verifies the final local
snapshot. To check the cache without downloading:

```bash
/home/test1/habanalabs-venv-optimum/bin/python download_hf_models.py \
  --all-defaults \
  --verify-only
```

## Chat UI

Start the 235B tensor-parallel chat server bound to all interfaces. The server
uses port `8000` by default:

```bash
./start_chat_server.sh
```

By default, `chat_server.py` loads models from the already-downloaded local
Hugging Face snapshot and does not download missing files at startup. To allow
the server to download from Hugging Face, set `CHAT_MODEL_LOCAL_FILES_ONLY=0`.
If `HF_HOME` is not set, the server also uses this repository's `./hf_cache`
directory by default.

Then open `http://<server-ip>:8000/`.

You can override the default with `SERVER_PORT` or `--port`.

The script defaults to 8 Gaudi HPU processes. Override settings with
environment variables:

```bash
SERVER_PORT=8080 CHAT_TENSOR_PARALLEL_SIZE=8 ./start_chat_server.sh
```

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

The chat server exposes only `Qwen/Qwen3-32B` and `Qwen/Qwen3-235B-A22B` in the
model selector.

## vLLM Gaudi inference

Build vLLM and the Intel Gaudi hardware plugin from source. The script follows
the `vllm-project/vllm-gaudi` flow: it resolves the plugin's last verified vLLM
commit, installs vLLM with `VLLM_TARGET_DEVICE=empty`, then installs
`vllm-gaudi`.

```bash
PYTHON_BIN=/home/test1/habanalabs-venv/bin/python ./scripts/build_vllm_gaudi.sh
```

Start the CLI chat bot with a selected model. The model is loaded once, then
each chat turn prints TTFT and TPS metrics:

```bash
MODEL_ID=Qwen/Qwen3-235B-A22B \
VLLM_TP_SIZE=8 \
./start_vllm_gaudi_infer.sh
```

Inside the chat, use `/reset` to clear history and `/exit` to quit. To run a
single prompt and exit, pass `--once`:

```bash
MODEL_ID=Qwen/Qwen3-32B \
PROMPT="日本語で短く自己紹介して" \
./start_vllm_gaudi_infer.sh --once
```

The vLLM CLI defaults to local Hugging Face snapshots under `./hf_cache`. To let
vLLM download missing model files, pass `--no-local-files-only`:

```bash
MODEL_ID=Qwen/Qwen3-32B ./start_vllm_gaudi_infer.sh --no-local-files-only
```

## Performance notes

The built-in FastAPI server still uses the Transformers tensor-parallel path.
Use `vllm_gaudi_infer.py` for the first vLLM Gaudi inference path; the UI server
can be switched to vLLM after this path is validated on the target model.

The server enables Habana inference settings, int32 token inputs, and KV cache
explicitly, but the main remaining bottleneck is the single-HPU Transformers
execution path.

```bash
HF_HOME=$PWD/hf_cache /home/test1/habanalabs-venv/bin/python chat_server.py \
  --host 0.0.0.0 \
  --model-id Qwen/Qwen3-32B
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

This venv supports Qwen3 models such as `Qwen/Qwen3-32B`.

`Qwen/Qwen3-235B-A22B` is available as a Qwen3 causal LM option. The current demo
uses a single-HPU Transformers placement, so this checkpoint requires enough HPU
memory for the full model or a future multi-HPU placement path before it can be
used reliably.

`Qwen/Qwen3-32B-FP8` is accepted by the CLI, but the current HPU + Optimum Habana
FP8 path creates an FP8 KV cache while the model produces bf16 key/value states.
The demo therefore falls back to `Qwen/Qwen3-32B` for correct generation.
