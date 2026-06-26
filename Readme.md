# Gaudi demo

Run Qwen models on Intel Gaudi HPU. The chat server defaults to
`Qwen/Qwen3.6-27B`.

## Chat UI

Start a web chat server bound to all interfaces:

```bash
HF_HOME=$PWD/hf_cache /home/test1/habanalabs-venv/bin/python chat_server.py \
  --host 0.0.0.0 \
  --port 8000
```

Then open `http://<server-ip>:8000/`.

The first screen asks for a user name. Each user gets a separate chat screen and
history, saved in `chat_history.json`.

The UI can switch between:

- `Qwen/Qwen3.6-27B`
- `Qwen/Qwen3.6-35B-A3B-FP8`
- `Qwen/Qwen3-32B`

The reasoning strength selector changes the speed/depth tradeoff:

- `Low`: shortest responses, fastest, thinking disabled
- `Medium`: standard responses, thinking disabled
- `High`: longer responses, thinking enabled

The agent mode selector is applied per message:

- `Chat`: answer with the selected model only
- `Web検索`: search DuckDuckGo once, then answer with sources
- `Deep search`: run multiple DuckDuckGo searches, then answer with broader source context

DuckDuckGo is the default search engine. To temporarily switch back to Bing,
start the server with `SEARCH_ENGINE=bing`.

While a message is running, the UI shows agent steps such as web search,
source preparation, prompt construction, model generation, and completion.
Use the cancel button to stop the active request; the server marks the request as
cancelled and asks generation to stop at the next token boundary.

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
  --port 8000 \
  --model-id Qwen/Qwen3.6-27B
```

## CLI

```bash
HF_HOME=$PWD/hf_cache /home/test1/habanalabs-venv/bin/python run_qwen36_hpu.py \
  --prompt "日本語で短く自己紹介して"
```

This machine has Habana software 1.24 installed. The working Python environment is
`/home/test1/habanalabs-venv`.

Note: `Qwen/Qwen3.6-27B` requires a newer Transformers release with `qwen3_5`
support. `optimum-habana==1.21.0` is installed in the environment, but its public
wrappers are currently pinned to `transformers<4.56`, so this demo uses the
Habana PyTorch bridge directly while keeping the Gaudi/Optimum Habana environment
installed.
