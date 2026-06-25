#!/usr/bin/env python3
import argparse
import gc
import html
import importlib.metadata
import os
import re
import sys
import threading
import time
import base64
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

import torch
import habana_frameworks.torch.core as htcore
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, TextIteratorStreamer


DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
MODEL_SPECS = {
    "Qwen/Qwen3.6-35B-A3B-FP8": {
        "label": "Qwen3.6 35B A3B FP8",
        "kind": "image_text",
        "precision": "fp8 pretrained",
    },
    "Qwen/Qwen3.6-27B": {
        "label": "Qwen3.6 27B",
        "kind": "image_text",
        "precision": "bf16",
    },
    "Qwen/Qwen3-32B": {
        "label": "Qwen3 32B",
        "kind": "causal_lm",
        "precision": "bf16",
    },
}
REASONING_PRESETS = {
    "low": {
        "label": "Low",
        "enable_thinking": False,
        "max_new_tokens": 64,
    },
    "medium": {
        "label": "Medium",
        "enable_thinking": False,
        "max_new_tokens": 128,
    },
    "high": {
        "label": "High",
        "enable_thinking": True,
        "max_new_tokens": 512,
    },
}
AGENT_MODES = {
    "chat": {"label": "Chat", "description": "モデルだけで応答"},
    "web": {"label": "Web検索", "description": "Web検索結果を参照して応答"},
    "deep": {"label": "Deep search", "description": "複数検索で広めに調べて応答"},
}


HTML = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gaudi Qwen Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #171717;
      --muted: #6b6f76;
      --line: #deded8;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --assistant: #eef5f2;
      --user: #e8eefb;
      --shadow: 0 12px 36px rgba(30, 36, 46, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      max-width: 1080px;
      margin: 0 auto;
      padding: 18px;
      gap: 14px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      border-bottom: 1px solid var(--line);
      padding: 8px 0 16px;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 720;
      letter-spacing: 0;
    }

    .status {
      min-height: 34px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 0 12px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #9ca3af;
    }

    .dot.ready { background: #16a34a; }
    .dot.busy { background: #f59e0b; }

    main {
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    #messages {
      flex: 1;
      min-height: 420px;
      overflow-y: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .message {
      max-width: min(760px, 92%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      line-height: 1.58;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 15px;
    }

    .message.user {
      align-self: flex-end;
      background: var(--user);
    }

    .message.assistant {
      align-self: flex-start;
      background: var(--assistant);
    }

    .message.system {
      align-self: center;
      max-width: 680px;
      background: #fafafa;
      color: var(--muted);
      text-align: center;
      font-size: 14px;
    }

    .metrics {
      align-self: flex-start;
      color: var(--muted);
      font-size: 12px;
      margin-top: -8px;
      padding-left: 4px;
    }

    form {
      border-top: 1px solid var(--line);
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(150px, 220px) minmax(120px, 150px) minmax(120px, 160px) 1fr auto;
      gap: 10px;
      align-items: end;
      background: #fbfbf9;
    }

    textarea {
      width: 100%;
      min-height: 52px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      line-height: 1.45;
      color: var(--ink);
      background: #fff;
    }

    button, select {
      height: 44px;
      border-radius: 6px;
      border: 1px solid var(--line);
      font: inherit;
      font-size: 14px;
    }

    button {
      padding: 0 16px;
      color: #fff;
      background: var(--accent);
      border-color: var(--accent);
      cursor: pointer;
      font-weight: 650;
    }

    button:hover { background: var(--accent-strong); }
    button:disabled { cursor: not-allowed; opacity: 0.62; }

    select {
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
    }

    @media (max-width: 720px) {
      .app { padding: 10px; }
      header { align-items: flex-start; flex-direction: column; }
      #messages { min-height: 360px; padding: 12px; }
      form { grid-template-columns: 1fr; }
      button, select { width: 100%; }
      .message { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Gaudi Qwen Chat</h1>
      <div class="status"><span id="dot" class="dot"></span><span id="statusText">接続確認中</span><span id="precisionText"></span></div>
    </header>

    <main>
      <div id="messages">
        <div class="message system">選択した Qwen モデルが Intel Gaudi HPU 上で応答します。</div>
      </div>

      <form id="chatForm">
        <select id="model" aria-label="model">
          <option value="Qwen/Qwen3.6-35B-A3B-FP8">Qwen3.6 35B A3B FP8</option>
          <option value="Qwen/Qwen3.6-27B">Qwen3.6 27B</option>
          <option value="Qwen/Qwen3-32B">Qwen3 32B</option>
        </select>
        <select id="reasoning" aria-label="reasoning strength">
          <option value="low">Low</option>
          <option value="medium" selected>Medium</option>
          <option value="high">High</option>
        </select>
        <select id="agentMode" aria-label="agent mode">
          <option value="chat" selected>Chat</option>
          <option value="web">Web検索</option>
          <option value="deep">Deep search</option>
        </select>
        <textarea id="prompt" autocomplete="off" placeholder="メッセージを入力" autofocus></textarea>
        <button id="send" type="submit">送信</button>
      </form>
    </main>
  </div>

  <script>
    const messagesEl = document.querySelector("#messages");
    const form = document.querySelector("#chatForm");
    const promptEl = document.querySelector("#prompt");
    const sendEl = document.querySelector("#send");
    const modelEl = document.querySelector("#model");
    const reasoningEl = document.querySelector("#reasoning");
    const agentModeEl = document.querySelector("#agentMode");
    const dotEl = document.querySelector("#dot");
    const statusText = document.querySelector("#statusText");
    const precisionText = document.querySelector("#precisionText");
    const history = [];

    function setStatus(text, state) {
      statusText.textContent = text;
      dotEl.className = `dot ${state || ""}`;
    }

    function addMessage(role, text) {
      const node = document.createElement("div");
      node.className = `message ${role}`;
      node.textContent = text;
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return node;
    }

    function addMetrics(data) {
      const node = document.createElement("div");
      node.className = "metrics";
      node.textContent = `${data.agent_mode.toUpperCase()} · ${data.reasoning_effort.toUpperCase()} · TTFT ${data.ttft_sec.toFixed(2)}s · TPS ${data.tokens_per_sec.toFixed(2)} tok/s · ${data.generated_tokens} tokens`;
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function addSources(sources) {
      if (!sources || sources.length === 0) return;
      const node = document.createElement("div");
      node.className = "metrics";
      node.textContent = `Sources: ${sources.map((source, index) => `[${index + 1}] ${source.title} ${source.url}`).join(" · ")}`;
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function refreshHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        setStatus(data.model_loaded ? "ready" : "loading", data.model_loaded ? "ready" : "busy");
        precisionText.textContent = data.precision ? `· ${data.precision.toUpperCase()}` : "";
        if (data.active_model_id) modelEl.value = data.active_model_id;
      } catch {
        setStatus("offline", "");
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const prompt = promptEl.value.trim();
      if (!prompt) return;

      promptEl.value = "";
      sendEl.disabled = true;
      setStatus("generating", "busy");
      addMessage("user", prompt);
      history.push({ role: "user", content: prompt });
      const modeLabel = agentModeEl.options[agentModeEl.selectedIndex].textContent;
      const assistantNode = addMessage("assistant", agentModeEl.value === "chat" ? "生成中..." : `${modeLabel} 中...`);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: history,
            model_id: modelEl.value,
            reasoning_effort: reasoningEl.value,
            agent_mode: agentModeEl.value
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "request failed");
        assistantNode.textContent = data.reply || "";
        addMetrics(data);
        addSources(data.sources);
        history.push({ role: "assistant", content: data.reply || "" });
        setStatus("ready", "ready");
      } catch (error) {
        assistantNode.textContent = `エラー: ${error.message}`;
        setStatus("error", "");
      } finally {
        sendEl.disabled = false;
        promptEl.focus();
      }
    });

    modelEl.addEventListener("change", () => {
      history.length = 0;
      messagesEl.innerHTML = "";
      addMessage("system", `${modelEl.value} に切り替えます。初回応答時にモデルをロードします。`);
      sendEl.disabled = true;
      setStatus("switching", "busy");
      fetch("/api/switch_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelEl.value })
      }).catch(() => {});
      const selectedModel = modelEl.value;
      const waitForModel = setInterval(async () => {
        try {
          const res = await fetch("/api/health");
          const data = await res.json();
          if (data.model_loaded && data.active_model_id === selectedModel) {
            clearInterval(waitForModel);
            sendEl.disabled = false;
            setStatus("ready", "ready");
          }
        } catch {}
      }, 3000);
    });

    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    refreshHealth();
    setInterval(refreshHealth, 10000);
  </script>
</body>
</html>
"""


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_id: str = DEFAULT_MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    agent_mode: Literal["chat", "web", "deep"] = "chat"
    max_new_tokens: int | None = Field(default=None, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.05, le=1.0)
    enable_thinking: bool | None = None


class SwitchModelRequest(BaseModel):
    model_id: str


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class ChatResponse(BaseModel):
    reply: str
    precision: str
    reasoning_effort: str
    agent_mode: str
    sources: list[SearchResult] = []
    effective_max_new_tokens: int
    enable_thinking: bool
    elapsed_sec: float
    ttft_sec: float
    tokens_per_sec: float
    generated_tokens: int


def normalize_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path == "/l/":
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    encoded = parse_qs(parsed.query).get("u")
    if parsed.netloc.endswith("bing.com") and encoded:
        value = encoded[0]
        if value.startswith("a1"):
            value = value[2:]
        try:
            padding = "=" * (-len(value) % 4)
            return base64.urlsafe_b64decode(value + padding).decode("utf-8")
        except Exception:
            return url
    return url


def strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def web_search(query: str, limit: int = 5) -> list[SearchResult]:
    url = f"https://r.jina.ai/http://www.bing.com/search?q={quote_plus(query)}"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12,
    )
    response.raise_for_status()
    body = response.text
    pattern = re.compile(
        r"\n\d+\.\s+## \[(?P<title>.*?)\]\((?P<url>.*?)\)\n\n(?P<snippet>.*?)(?=\n\d+\.\s+## |\Z)",
        re.DOTALL,
    )
    results = []
    seen = set()
    for match in pattern.finditer(body):
        result_url = normalize_duckduckgo_url(html.unescape(match.group("url")))
        if result_url in seen:
            continue
        seen.add(result_url)
        results.append(
            SearchResult(
                title=strip_tags(match.group("title")),
                url=result_url,
                snippet=strip_tags(match.group("snippet")),
            )
        )
        if len(results) >= limit:
            break
    return results


def search_queries(prompt: str, mode: str) -> list[str]:
    if mode == "web":
        return [prompt]
    return [
        prompt,
        f"{prompt} 最新",
        f"{prompt} 背景 解説",
    ]


def collect_sources(prompt: str, mode: str) -> list[SearchResult]:
    if mode == "chat":
        return []
    sources = []
    seen = set()
    per_query = 4 if mode == "deep" else 5
    max_sources = 8 if mode == "deep" else 5
    for query in search_queries(prompt, mode):
        for result in web_search(query, limit=per_query):
            if result.url in seen:
                continue
            seen.add(result.url)
            sources.append(result)
            if len(sources) >= max_sources:
                return sources
    return sources


def request_with_sources(request: ChatRequest, sources: list[SearchResult]) -> ChatRequest:
    if not sources:
        return request
    messages = list(request.messages)
    last = messages[-1]
    source_lines = "\n".join(
        f"[{index}] {source.title}\nURL: {source.url}\n概要: {source.snippet}"
        for index, source in enumerate(sources, start=1)
    )
    mode_name = AGENT_MODES[request.agent_mode]["label"]
    enriched = (
        f"{last.content}\n\n"
        f"{mode_name} の検索結果:\n{source_lines}\n\n"
        "上の検索結果を根拠として使い、必要なら [1] のように番号で出典を示して日本語で答えてください。"
    )
    messages[-1] = ChatMessage(role=last.role, content=enriched)
    return request.model_copy(update={"messages": messages})


class ChatEngine:
    def __init__(self, default_model_id: str) -> None:
        self.default_model_id = default_model_id
        self.active_model_id = None
        self.model_kind = None
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()
        self.loaded_at = None
        self.precision = "bf16"

    def load(self, model_id: str | None = None) -> None:
        model_id = model_id or self.default_model_id
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unsupported model: {model_id}")
        if self.is_loaded and self.active_model_id == model_id:
            return

        if not torch.hpu.is_available():
            raise RuntimeError("HPU is not available. Check Habana driver/runtime setup.")

        self.unload()
        spec = MODEL_SPECS[model_id]
        started = time.time()
        if spec["kind"] == "image_text":
            tokenizer = AutoProcessor.from_pretrained(model_id)
            model_cls = AutoModelForImageTextToText
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model_cls = AutoModelForCausalLM

        self.model = model_cls.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "hpu"},
        )
        self.model.eval()
        self.tokenizer = tokenizer
        self.active_model_id = model_id
        self.model_kind = spec["kind"]
        htcore.mark_step()
        torch.hpu.synchronize()
        self.precision = spec["precision"]
        self.loaded_at = time.time()
        print(f"Loaded {model_id} on HPU in {self.loaded_at - started:.1f}s ({self.precision})", flush=True)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
        self.model = None
        self.tokenizer = None
        self.active_model_id = None
        self.model_kind = None
        self.loaded_at = None
        self.precision = "bf16"
        gc.collect()
        if torch.hpu.is_available():
            try:
                torch.hpu.empty_cache()
                torch.hpu.synchronize()
            except Exception:
                pass

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def render_prompt(self, request: ChatRequest) -> str:
        assert self.tokenizer is not None
        if self.model_kind == "image_text":
            messages = [
                {"role": message.role, "content": [{"type": "text", "text": message.content}]}
                for message in request.messages
            ]
        else:
            messages = [{"role": message.role, "content": message.content} for message in request.messages]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking(request),
        )

    def tokenize(self, text: str) -> dict:
        assert self.tokenizer is not None
        if self.model_kind == "image_text":
            return self.tokenizer(text=[text], return_tensors="pt")
        return self.tokenizer([text], return_tensors="pt")

    def generate(self, request: ChatRequest, sources: list[SearchResult] | None = None) -> ChatResponse:
        sources = sources or []
        with self.lock:
            self.load(request.model_id)
            assert self.tokenizer is not None
            assert self.model is not None

            text = self.render_prompt(request)
            inputs = self.tokenize(text)
            inputs = {key: value.to("hpu") for key, value in inputs.items()}

            started = time.time()
            do_sample = request.temperature > 0
            streamer = TextIteratorStreamer(
                self.tokenizer.tokenizer if hasattr(self.tokenizer, "tokenizer") else self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generated_output = {}
            generation_error = {}

            def run_generate() -> None:
                try:
                    with torch.inference_mode():
                        generated_output["output_ids"] = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens(request),
                            do_sample=do_sample,
                            temperature=request.temperature if do_sample else None,
                            top_p=request.top_p if do_sample else None,
                            pad_token_id=self.eos_token_id(),
                            streamer=streamer,
                        )
                        htcore.mark_step()
                        torch.hpu.synchronize()
                except Exception as exc:
                    generation_error["error"] = exc

            first_token_at = None
            chunks = []
            worker = threading.Thread(target=run_generate)
            worker.start()
            for chunk in streamer:
                if first_token_at is None:
                    first_token_at = time.time()
                chunks.append(chunk)
            worker.join()

            if generation_error:
                raise generation_error["error"]

            finished = time.time()
            output_ids = generated_output["output_ids"]
            prompt_len = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, prompt_len:]
            generated_tokens = int(generated_ids.shape[-1])
            reply = self.clean_reply("".join(chunks))
            if not reply:
                reply = self.clean_reply(self.decode(generated_ids))
            if not reply:
                reply = "High の推論でトークン上限に達しました。Medium に下げるか、上限を増やしてください。"

            ttft_sec = (first_token_at or finished) - started
            decode_sec = max(finished - (first_token_at or started), 1e-9)
            tokens_per_sec = generated_tokens / decode_sec
            return ChatResponse(
                reply=reply,
                precision=self.precision,
                reasoning_effort=request.reasoning_effort,
                agent_mode=request.agent_mode,
                sources=sources,
                effective_max_new_tokens=self.max_new_tokens(request),
                enable_thinking=self.enable_thinking(request),
                elapsed_sec=finished - started,
                ttft_sec=ttft_sec,
                tokens_per_sec=tokens_per_sec,
                generated_tokens=generated_tokens,
            )

    def eos_token_id(self) -> int | None:
        assert self.tokenizer is not None
        tokenizer = self.tokenizer.tokenizer if hasattr(self.tokenizer, "tokenizer") else self.tokenizer
        return tokenizer.eos_token_id

    def decode(self, generated_ids) -> str:
        assert self.tokenizer is not None
        decoder = self.tokenizer.tokenizer if hasattr(self.tokenizer, "tokenizer") else self.tokenizer
        return decoder.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def clean_reply(self, text: str) -> str:
        text = text.strip()
        if "<think>" not in text:
            return text
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return ""

    def max_new_tokens(self, request: ChatRequest) -> int:
        preset_tokens = REASONING_PRESETS[request.reasoning_effort]["max_new_tokens"]
        return request.max_new_tokens or int(preset_tokens)

    def enable_thinking(self, request: ChatRequest) -> bool:
        if request.enable_thinking is not None:
            return request.enable_thinking
        return bool(REASONING_PRESETS[request.reasoning_effort]["enable_thinking"])


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def create_app(model_id: str) -> FastAPI:
    engine = ChatEngine(model_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Starting Gaudi Qwen chat server", flush=True)
        print(f"torch={package_version('torch')}", flush=True)
        print(f"habana-torch-plugin={package_version('habana-torch-plugin')}", flush=True)
        print(f"optimum-habana={package_version('optimum-habana')}", flush=True)
        print(f"transformers={package_version('transformers')}", flush=True)
        print("fp8=pretrained when selecting Qwen/Qwen3.6-35B-A3B-FP8", flush=True)
        engine.load(model_id)
        yield

    app = FastAPI(title="Gaudi Qwen Chat", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/health")
    def health() -> dict:
        return {
            "default_model_id": engine.default_model_id,
            "models": MODEL_SPECS,
            "reasoning_presets": REASONING_PRESETS,
            "agent_modes": AGENT_MODES,
            "active_model_id": engine.active_model_id,
            "model_loaded": engine.is_loaded,
            "precision": engine.precision,
            "fp8_enabled": engine.precision.startswith("fp8"),
            "hpu_available": torch.hpu.is_available(),
            "hpu_devices": torch.hpu.device_count() if torch.hpu.is_available() else 0,
            "loaded_at": engine.loaded_at,
        }

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            if engine.is_loaded and engine.active_model_id != request.model_id:
                raise HTTPException(
                    status_code=409,
                    detail="Select the model in the UI first. The server restarts to switch models cleanly.",
                )
            sources = collect_sources(request.messages[-1].content, request.agent_mode)
            enriched_request = request_with_sources(request, sources)
            return engine.generate(enriched_request, sources=sources)
        except HTTPException:
            raise
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Web search failed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/switch_model")
    def switch_model(request: SwitchModelRequest) -> dict:
        if request.model_id not in MODEL_SPECS:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model_id}")
        if engine.is_loaded and engine.active_model_id == request.model_id:
            return {"restarting": False, "active_model_id": engine.active_model_id}

        def restart() -> None:
            env = os.environ.copy()
            env["MODEL_ID"] = request.model_id
            env["SERVER_HOST"] = SERVER_HOST
            env["SERVER_PORT"] = str(SERVER_PORT)
            args = [
                sys.executable,
                os.path.abspath(__file__),
                "--host",
                SERVER_HOST,
                "--port",
                str(SERVER_PORT),
                "--model-id",
                request.model_id,
            ]
            os.execve(sys.executable, args, env)

        threading.Timer(0.5, restart).start()
        return {"restarting": True, "target_model_id": request.model_id}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a web chat UI for Qwen on Gaudi.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, choices=sorted(MODEL_SPECS))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


app = create_app(os.environ.get("MODEL_ID", DEFAULT_MODEL_ID))


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
    app = create_app(args.model_id)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
