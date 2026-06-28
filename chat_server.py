#!/usr/bin/env python3
import argparse
import gc
import html
import importlib.metadata
import json
import os
import queue
import re
import sys
import threading
import time
import base64
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

# These must be set before importing torch / habana_frameworks. Lazy mode can be
# faster for some LLMs, but this model path currently starts more reliably in
# eager mode. Weight sharing is also disabled by default to avoid extra HPU
# memory consumption.
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
os.environ.setdefault("PT_HPU_WEIGHT_SHARING", "0")

import torch
import habana_frameworks.torch.core as htcore
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


COMPAT_DEFAULT_MODEL_ID = "Qwen/Qwen3-32B"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HF_HOME = PROJECT_ROOT / "hf_cache"


def transformer_supports_model_type(model_type: str) -> bool:
    return model_type in CONFIG_MAPPING


DEFAULT_MODEL_ID = COMPAT_DEFAULT_MODEL_ID
DEFAULT_SERVER_PORT = 8000
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", DEFAULT_SERVER_PORT))
HISTORY_PATH = Path(os.environ.get("CHAT_HISTORY_PATH", "chat_history.json"))
SEARCH_ENGINE = os.environ.get("SEARCH_ENGINE", "duckduckgo")
SEARCH_TIMEOUT_SEC = float(os.environ.get("SEARCH_TIMEOUT_SEC", "8"))
AUTO_SEARCH_DEFAULT = os.environ.get("AUTO_SEARCH_DEFAULT", "1") == "1"
HPU_EXECUTION_MODE = "lazy" if os.environ.get("PT_HPU_LAZY_MODE") == "1" else "eager"
USE_INT32_INPUTS = os.environ.get("USE_INT32_INPUTS", "1") == "1"
MODEL_PLACEMENT = "single_hpu_transformers"
OPTIMUM_HABANA_ENABLED: bool | None = None
MODEL_LOCAL_FILES_ONLY = os.environ.get("CHAT_MODEL_LOCAL_FILES_ONLY", "1") != "0"
HF_MODEL_REVISION = os.environ.get("HF_MODEL_REVISION", "main")
MODEL_SPECS = {
    "Qwen/Qwen3-32B": {
        "label": "Qwen3 32B",
        "kind": "causal_lm",
        "precision": "bf16",
    },
    "Qwen/Qwen3-235B-A22B": {
        "label": "Qwen3 235B A22B",
        "kind": "causal_lm",
        "precision": "bf16",
    },
}
MODEL_REQUIRED_TYPES = {}
FP8_CORRECTNESS_FALLBACKS = {}


def is_model_supported(model_id: str) -> bool:
    required_type = MODEL_REQUIRED_TYPES.get(model_id)
    return required_type is None or transformer_supports_model_type(required_type)


def supported_model_specs() -> dict[str, dict[str, str]]:
    return {model_id: spec for model_id, spec in MODEL_SPECS.items() if is_model_supported(model_id)}


def is_fp8_model(model_id: str) -> bool:
    return model_id.upper().endswith("-FP8")


def resolve_execution_model_id(model_id: str) -> str:
    if torch.hpu.is_available() and model_id in FP8_CORRECTNESS_FALLBACKS:
        fallback_model_id = FP8_CORRECTNESS_FALLBACKS[model_id]
        print(
            f"{model_id} produces non-finite logits on the current HPU Transformers FP8 path; "
            f"using {fallback_model_id} for correctness.",
            flush=True,
        )
        return fallback_model_id
    return model_id


def hf_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return DEFAULT_HF_HOME / "hub"


def hf_model_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def local_snapshot_path(model_id: str, revision: str = HF_MODEL_REVISION) -> Path | None:
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


def resolve_pretrained_source(model_id: str) -> tuple[str, dict[str, object]]:
    if not MODEL_LOCAL_FILES_ONLY:
        return model_id, {"revision": HF_MODEL_REVISION}

    snapshot_path = local_snapshot_path(model_id)
    if snapshot_path is None:
        raise RuntimeError(
            f"Local snapshot for {model_id} was not found in {hf_cache_root()}. "
            "Download it first with: "
            f"{sys.executable} download_hf_models.py "
            f"--model-id {model_id} --prepare"
        )

    print(f"Using local Hugging Face snapshot: {snapshot_path}", flush=True)
    return str(snapshot_path), {"local_files_only": True}


REASONING_PRESETS = {
    "low": {
        "label": "Low",
        "enable_thinking": False,
        "max_new_tokens": 128,
    },
    "medium": {
        "label": "Medium",
        "enable_thinking": False,
        "max_new_tokens": 512,
    },
    "high": {
        "label": "High",
        "enable_thinking": True,
        "max_new_tokens": 1024,
    },
}
AGENT_MODES = {
    "auto": {"label": "Auto", "description": "必要な場合だけ自動検索して応答"},
    "chat": {"label": "Chat", "description": "モデルだけで応答"},
    "web": {"label": "Web検索", "description": "Web検索結果を参照して応答"},
    "deep": {"label": "Deep search", "description": "複数検索で広めに調べて応答"},
}
USER_AGENT_MODES = {
    key: value for key, value in AGENT_MODES.items() if key in {"auto", "chat", "deep"}
}
AGENT_MODE_MIN_TOKENS = {
    "deep": 4096,
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
      max-width: 1520px;
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

    .login-view {
      min-height: 420px;
      display: grid;
      place-items: center;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 24px;
    }

    .login-panel {
      width: min(420px, 100%);
      display: grid;
      gap: 14px;
    }

    .login-title {
      font-size: 18px;
      font-weight: 720;
    }

    .login-error {
      min-height: 20px;
      color: #b91c1c;
      font-size: 13px;
    }

    .sessionbar {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfbf9;
    }

    .session-user {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .workspace {
      min-height: 0;
      display: grid;
      grid-template-columns: 240px minmax(0, 1fr);
      gap: 14px;
    }

    .threadbar {
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 10px;
      align-content: start;
      min-height: 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .thread-current {
      display: grid;
      gap: 2px;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
    }

    .thread-current strong {
      color: var(--ink);
      font-size: 14px;
      overflow-wrap: anywhere;
    }

    .thread-list {
      display: grid;
      gap: 8px;
      overflow-y: auto;
      min-height: 0;
    }

    .thread-item {
      min-height: 38px;
      width: 100%;
      display: inline-grid;
      align-items: center;
      justify-content: start;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfbf9;
      color: var(--ink);
      padding: 0 10px;
      font-size: 13px;
      cursor: pointer;
      max-width: 220px;
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-decoration: none;
    }

    .thread-item.active {
      border-color: var(--accent);
      background: #e6f3f0;
      font-weight: 700;
    }

    .activitybar {
      display: grid;
      grid-template-columns: minmax(110px, auto) 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      background: #f7faf9;
      color: var(--muted);
      font-size: 13px;
    }

    .activity-mode {
      color: var(--ink);
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .activity-detail {
      min-width: 0;
      overflow-wrap: anywhere;
    }

    .activity-elapsed {
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }

    .activitybar.busy .activity-mode::before {
      content: "";
      width: 8px;
      height: 8px;
      display: inline-block;
      margin-right: 7px;
      border-radius: 999px;
      background: #f59e0b;
      vertical-align: 1px;
    }

    .activitybar.ready .activity-mode::before {
      content: "";
      width: 8px;
      height: 8px;
      display: inline-block;
      margin-right: 7px;
      border-radius: 999px;
      background: #16a34a;
      vertical-align: 1px;
    }

    @media (max-width: 720px) {
      .workspace {
        grid-template-columns: 1fr;
      }

      .activitybar {
        grid-template-columns: 1fr;
      }

      .threadbar {
        max-height: 220px;
      }

      .activity-elapsed {
        white-space: normal;
      }
    }

    .field {
      display: grid;
      gap: 5px;
    }

    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    input {
      width: 100%;
      height: 44px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 12px;
      font: inherit;
      color: var(--ink);
      background: #fff;
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

    .hidden { display: none !important; }

    #messages {
      flex: 1;
      min-height: min(68vh, 720px);
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

    .steps {
      align-self: flex-start;
      width: min(760px, 92%);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfbf9;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
      display: grid;
      gap: 7px;
    }

    .step {
      display: grid;
      grid-template-columns: 18px 1fr;
      gap: 8px;
      align-items: start;
      line-height: 1.35;
    }

    .step-marker {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-top: 4px;
      background: #c8ccd2;
    }

    .step.active .step-marker { background: #f59e0b; }
    .step.done .step-marker { background: #16a34a; }
    .step.error .step-marker { background: #dc2626; }

    .step-title {
      color: var(--ink);
      font-weight: 640;
    }

    .step-detail {
      font-size: 12px;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }

    #chatForm {
      border-top: 1px solid var(--line);
      padding: 14px;
      display: grid;
      grid-template-columns: minmax(220px, 300px) minmax(0, 1fr) auto auto;
      gap: 10px;
      align-items: end;
      background: #fbfbf9;
    }

    .option-panel {
      display: grid;
      gap: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }

    .option-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .option-controls {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }

    textarea {
      width: 100%;
      min-height: 76px;
      max-height: 240px;
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

    button.cancel {
      background: #b91c1c;
      border-color: #b91c1c;
    }

    button.cancel:hover { background: #991b1b; }

    button.secondary,
    a.secondary {
      color: var(--ink);
      background: #fff;
      border-color: var(--line);
    }

    button.secondary:hover,
    a.secondary:hover {
      background: #f2f3f0;
    }

    .action-link {
      height: 44px;
      display: inline-grid;
      place-items: center;
      padding: 0 16px;
      border: 1px solid var(--line);
      border-radius: 6px;
      text-decoration: none;
      font-weight: 650;
    }

    select {
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
    }

    @media (max-width: 720px) {
      .app { padding: 10px; }
      header { align-items: flex-start; flex-direction: column; }
      .sessionbar { grid-template-columns: 1fr; }
      #messages { min-height: 360px; padding: 12px; }
      #chatForm { grid-template-columns: 1fr; }
      .option-controls { grid-template-columns: 1fr; }
      button, select { width: 100%; }
      .message { max-width: 100%; }
    }
  </style>
</head>
<body data-initial-user-id="" data-initial-display-name="">
  <div class="app">
    <header>
      <h1>Gaudi Qwen Chat</h1>
      <div class="status"><span id="dot" class="dot"></span><span id="statusText">接続確認中</span><span id="precisionText"></span></div>
    </header>

    <section id="loginView" class="login-view">
      <form id="loginForm" class="login-panel" action="/login" method="get">
        <div class="login-title">ログイン</div>
        <div class="field">
          <label for="loginName">ユーザー名</label>
          <input id="loginName" name="display_name" autocomplete="username" placeholder="例: kazuki" autofocus />
        </div>
        <button id="loginButton" type="submit">ログイン</button>
        <div id="loginError" class="login-error"></div>
      </form>
    </section>

    <div id="chatView" class="workspace hidden">
      <aside class="threadbar">
        <div class="thread-current">現在のスレッド<strong id="currentThreadTitle">メイン</strong></div>
        <a id="newThread" class="secondary action-link" href="/threads/new">新規スレッド</a>
        <button id="deleteThread" class="secondary" type="button">削除</button>
        <div id="threadList" class="thread-list"></div>
      </aside>

      <main>
        <div class="sessionbar">
          <div class="session-user">ログイン中: <strong id="currentUserName"></strong></div>
          <button id="clearHistory" class="secondary" type="button">履歴削除</button>
          <a id="logout" class="secondary action-link" href="/logout">ログアウト</a>
        </div>

        <div id="activityBar" class="activitybar ready">
          <div id="activityMode" class="activity-mode">待機中</div>
          <div id="activityDetail" class="activity-detail">メッセージを送信できます</div>
          <div id="activityElapsed" class="activity-elapsed"></div>
        </div>

        <div id="messages">
          <div class="message system">選択した Qwen モデルが Intel Gaudi HPU 上で応答します。</div>
        </div>

        <form id="chatForm" action="/chat/send" method="post">
          <input id="threadId" name="thread_id" type="hidden" value="default" />
          <div class="option-panel">
            <div class="option-title">オプション</div>
            <div class="option-controls">
              <select id="model" name="model_id" aria-label="model">
                <option value="Qwen/Qwen3-32B" selected>Qwen3 32B</option>
                <option value="Qwen/Qwen3-235B-A22B">Qwen3 235B A22B</option>
              </select>
              <select id="reasoning" name="reasoning_effort" aria-label="reasoning strength">
                <option value="low">Low</option>
                <option value="medium" selected>Medium</option>
                <option value="high">High</option>
              </select>
              <select id="agentMode" name="agent_mode" aria-label="agent mode">
                <option value="auto" selected>Auto</option>
                <option value="chat">Chat</option>
                <option value="deep">Deep search</option>
              </select>
            </div>
          </div>
          <textarea id="prompt" name="prompt" autocomplete="off" placeholder="メッセージを入力" autofocus></textarea>
          <button id="send" type="submit">送信</button>
          <button id="cancelJob" class="cancel hidden" type="button">キャンセル</button>
        </form>
      </main>
    </div>
  </div>

  <script>
    const messagesEl = document.querySelector("#messages");
    const form = document.querySelector("#chatForm");
    const promptEl = document.querySelector("#prompt");
    const sendEl = document.querySelector("#send");
    const cancelJobEl = document.querySelector("#cancelJob");
    const modelEl = document.querySelector("#model");
    const reasoningEl = document.querySelector("#reasoning");
    const agentModeEl = document.querySelector("#agentMode");
    const loginViewEl = document.querySelector("#loginView");
    const chatViewEl = document.querySelector("#chatView");
    const loginFormEl = document.querySelector("#loginForm");
    const loginNameEl = document.querySelector("#loginName");
    const loginErrorEl = document.querySelector("#loginError");
    const currentUserNameEl = document.querySelector("#currentUserName");
    const currentThreadTitleEl = document.querySelector("#currentThreadTitle");
    const threadListEl = document.querySelector("#threadList");
    const newThreadEl = document.querySelector("#newThread");
    const deleteThreadEl = document.querySelector("#deleteThread");
    const threadIdEl = document.querySelector("#threadId");
    const clearHistoryEl = document.querySelector("#clearHistory");
    const logoutEl = document.querySelector("#logout");
    const dotEl = document.querySelector("#dot");
    const statusText = document.querySelector("#statusText");
    const precisionText = document.querySelector("#precisionText");
    const activityBarEl = document.querySelector("#activityBar");
    const activityModeEl = document.querySelector("#activityMode");
    const activityDetailEl = document.querySelector("#activityDetail");
    const activityElapsedEl = document.querySelector("#activityElapsed");
    const chatHistory = [];
    const defaultSystemMessage = "選択した Qwen モデルが Intel Gaudi HPU 上で応答します。";
    const activeRequestIds = new Set();
    let activeRequestId = null;
    let activeController = null;
    let isRunning = false;
    let activityStartedAt = null;
    let activityTimer = null;
    const initialUserId = document.body.dataset.initialUserId || "";
    const initialDisplayName = document.body.dataset.initialDisplayName || "";
    const initialThreadId = document.body.dataset.initialThreadId || "default";
    let currentUserId = sessionStorage.getItem("gaudiChatUserId") || initialUserId;
    let currentDisplayName = sessionStorage.getItem("gaudiChatDisplayName") || initialDisplayName;
    let currentThreadId = sessionStorage.getItem("gaudiChatThreadId") || initialThreadId || "default";
    let threadListSignature = "";
    if (initialUserId && !sessionStorage.getItem("gaudiChatUserId")) {
      sessionStorage.setItem("gaudiChatUserId", initialUserId);
      sessionStorage.setItem("gaudiChatDisplayName", initialDisplayName || initialUserId);
    }

    async function fetchJson(url, options = {}, timeoutMs = 5000) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const requestOptions = Object.assign({}, options, { signal: options.signal || controller.signal });
        const res = await fetch(url, requestOptions);
        const data = await res.json();
        return { res, data };
      } finally {
        clearTimeout(timeout);
      }
    }

    async function cancelActiveRequest() {
      if (!activeRequestId) return;
      cancelJobEl.disabled = true;
      cancelJobEl.textContent = "キャンセル中";
      setStatus("cancelling", "busy");
      try {
        await fetch(`/api/cancel/${encodeURIComponent(activeRequestId)}`, { method: "POST" });
      } catch (_error) {}
      if (activeController) activeController.abort();
    }

    function updateCancelButton() {
      const requestIds = Array.from(activeRequestIds);
      activeRequestId = requestIds.length ? requestIds[requestIds.length - 1] : null;
      cancelJobEl.classList.toggle("hidden", !activeRequestId);
      cancelJobEl.disabled = false;
      cancelJobEl.textContent = "キャンセル";
    }

    function trackActiveRequest(requestId) {
      activeRequestIds.add(requestId);
      updateCancelButton();
    }

    function untrackActiveRequest(requestId) {
      activeRequestIds.delete(requestId);
      updateCancelButton();
    }

    function setStatus(text, state) {
      statusText.textContent = text;
      dotEl.className = `dot ${state || ""}`;
    }

    function updateActivity(mode, detail, state) {
      activityModeEl.textContent = mode || "待機中";
      activityDetailEl.textContent = detail || "";
      activityBarEl.className = `activitybar ${state || "ready"}`;
    }

    function startActivity(mode, detail) {
      activityStartedAt = Date.now();
      updateActivity(mode, detail, "busy");
      if (activityTimer) clearInterval(activityTimer);
      activityTimer = setInterval(() => {
        if (!activityStartedAt) return;
        const elapsed = Math.floor((Date.now() - activityStartedAt) / 1000);
        activityElapsedEl.textContent = `${elapsed}s`;
      }, 1000);
      activityElapsedEl.textContent = "0s";
    }

    function finishActivity(detail) {
      if (activityTimer) clearInterval(activityTimer);
      activityTimer = null;
      activityStartedAt = null;
      activityElapsedEl.textContent = "";
      updateActivity("待機中", detail || "メッセージを送信できます", "ready");
    }

    function activeStepSummary(steps) {
      if (!steps || steps.length === 0) return "";
      const active = steps.find((step) => step.status === "active");
      if (active) return active.detail ? `${active.label}: ${active.detail}` : active.label;
      const lastDone = [...steps].reverse().find((step) => step.status === "done");
      if (lastDone) return lastDone.detail ? `${lastDone.label}: ${lastDone.detail}` : lastDone.label;
      return "";
    }

    function addMessage(role, text) {
      const node = document.createElement("div");
      node.className = `message ${role}`;
      node.textContent = text;
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return node;
    }

    function renderConversation(messages) {
      chatHistory.length = 0;
      messagesEl.innerHTML = "";
      addMessage("system", defaultSystemMessage);
      for (const message of messages || []) {
        if (message.role !== "user" && message.role !== "assistant") continue;
        chatHistory.push({ role: message.role, content: message.content });
        addMessage(message.role, message.content);
      }
    }

    function renderThreadList(threads) {
      threadListEl.innerHTML = "";
      for (const thread of threads || []) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `thread-item${thread.thread_id === currentThreadId ? " active" : ""}`;
        button.textContent = `${thread.running ? "生成中 · " : ""}${thread.title || "無題"}`;
        button.title = `${thread.title || "無題"} · ${thread.message_count || 0}件`;
        button.addEventListener("click", () => loadThread(thread.thread_id));
        threadListEl.appendChild(button);
      }
    }

    async function loadThreads() {
      if (!currentUserId) return;
      const { res, data } = await fetchJson(`/api/threads/${encodeURIComponent(currentUserId)}`);
      if (!res.ok) throw new Error(data.detail || "スレッド一覧を読み込めませんでした");
      if (data.length > 0 && !data.some((thread) => thread.thread_id === currentThreadId)) {
        currentThreadId = data[0].thread_id;
      }
      const signature = JSON.stringify((data || []).map((thread) => [
        thread.thread_id,
        thread.title,
        thread.message_count,
        thread.updated_at,
        thread.running,
        thread.thread_id === currentThreadId
      ]));
      if (signature !== threadListSignature) {
        threadListSignature = signature;
        renderThreadList(data);
      }
    }

    async function loadThread(threadId) {
      if (!currentUserId) return;
      const { res, data } = await fetchJson(
        `/api/threads/${encodeURIComponent(currentUserId)}/${encodeURIComponent(threadId)}`
      );
      if (!res.ok) throw new Error(data.detail || "スレッドを読み込めませんでした");
      currentThreadId = data.thread_id;
      sessionStorage.setItem("gaudiChatThreadId", currentThreadId);
      threadIdEl.value = currentThreadId;
      currentThreadTitleEl.textContent = data.title || "無題";
      renderConversation(data.messages);
      await loadThreads();
      showChat();
    }

    async function createThread() {
      if (!currentUserId || isRunning) return;
      const { res, data } = await fetchJson("/api/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: currentUserId })
      });
      if (!res.ok) throw new Error(data.detail || "スレッドを作成できませんでした");
      await loadThread(data.thread_id);
      finishActivity("新しいスレッドを作成しました");
    }

    async function deleteCurrentThread() {
      if (!currentUserId || !currentThreadId || isRunning) return;
      if (currentThreadId === "default") {
        addMessage("system", "メインスレッドは削除できません。履歴削除を使ってください。");
        return;
      }
      if (!window.confirm("このスレッドを削除しますか？")) return;
      const { res, data } = await fetchJson(
        `/api/threads/${encodeURIComponent(currentUserId)}/${encodeURIComponent(currentThreadId)}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error(data.detail || "スレッドを削除できませんでした");
      currentThreadId = "default";
      sessionStorage.setItem("gaudiChatThreadId", currentThreadId);
      await loadThread(currentThreadId);
      finishActivity("スレッドを削除しました");
    }

    function showLogin() {
      chatViewEl.classList.add("hidden");
      loginViewEl.classList.remove("hidden");
      loginNameEl.value = currentDisplayName || "";
      loginNameEl.focus();
    }

    function showChat() {
      loginViewEl.classList.add("hidden");
      chatViewEl.classList.remove("hidden");
      currentUserNameEl.textContent = currentDisplayName || currentUserId;
      promptEl.focus();
    }

    async function loadHistory(userId) {
      await loadThreads();
      const { res, data } = await fetchJson(
        `/api/threads/${encodeURIComponent(userId)}/${encodeURIComponent(currentThreadId)}`
      );
      if (!res.ok) throw new Error(data.detail || "履歴を読み込めませんでした");
      renderConversation(data.messages);
      currentDisplayName = data.display_name;
      currentThreadId = data.thread_id;
      threadIdEl.value = currentThreadId;
      currentUserNameEl.textContent = currentDisplayName;
      currentThreadTitleEl.textContent = data.title || "メイン";
      await loadThreads();
    }

    async function login(displayName) {
      const { res, data } = await fetchJson("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: displayName })
      });
      if (!res.ok) throw new Error(data.detail || "ログインできませんでした");
      currentUserId = data.user_id;
      currentDisplayName = data.display_name;
      sessionStorage.setItem("gaudiChatUserId", currentUserId);
      sessionStorage.setItem("gaudiChatDisplayName", currentDisplayName);
      currentThreadId = "default";
      sessionStorage.setItem("gaudiChatThreadId", currentThreadId);
      renderConversation([]);
      showChat();
      setStatus("ready", "ready");
      finishActivity("ログインしました");
      try {
        await loadHistory(currentUserId);
        showChat();
      } catch (error) {
        addMessage("system", `履歴を読み込めませんでした: ${error.message}`);
      }
    }

    function addMetrics(data) {
      const node = document.createElement("div");
      node.className = "metrics";
      const mode = data.resolved_mode ? `${data.agent_mode.toUpperCase()}→${data.resolved_mode.toUpperCase()}` : data.agent_mode.toUpperCase();
      node.textContent = `${mode} · ${data.reasoning_effort.toUpperCase()} · TTFT ${data.ttft_sec.toFixed(2)}s · TPS ${data.tokens_per_sec.toFixed(2)} tok/s · ${data.generated_tokens} tokens`;
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

    function addStepPanel() {
      const node = document.createElement("div");
      node.className = "steps";
      node.textContent = "準備中...";
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return node;
    }

    function clientInitialSteps(mode) {
      const steps = [{ label: "リクエスト受付", status: "done", detail: mode.toUpperCase() }];
      if (mode === "auto") {
        steps.push({ label: "検索要否判定", status: "active", detail: "検索が必要か判定しています" });
      }
      if (mode === "deep") {
        steps.push({ label: "Deep search", status: "active", detail: "検索開始を待っています" });
        steps.push({ label: "検索結果の整理", status: "pending", detail: "" });
      }
      steps.push({ label: "プロンプト作成", status: "pending", detail: "" });
      steps.push({ label: "モデル生成", status: "pending", detail: "" });
      steps.push({ label: "応答完了", status: "pending", detail: "" });
      return steps;
    }

    function renderSteps(node, steps) {
      if (!steps || steps.length === 0) {
        node.textContent = "準備中...";
        return;
      }
      node.textContent = "";
      for (const step of steps) {
        const row = document.createElement("div");
        row.className = `step ${step.status}`;
        const marker = document.createElement("span");
        marker.className = "step-marker";
        const body = document.createElement("div");
        const title = document.createElement("div");
        title.className = "step-title";
        title.textContent = step.label;
        body.appendChild(title);
        if (step.detail) {
          const detail = document.createElement("div");
          detail.className = "step-detail";
          detail.textContent = step.detail;
          body.appendChild(detail);
        }
        row.appendChild(marker);
        row.appendChild(body);
        node.appendChild(row);
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function makeRequestId() {
      if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
      return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function syncModelOptions(data) {
      const selectedModel = data.active_model_id || data.default_model_id || modelEl.value;
      const models = data.models || {};
      const orderedModelIds = Object.keys(models).sort((left, right) => {
        if (left === selectedModel) return -1;
        if (right === selectedModel) return 1;
        if (left === data.default_model_id) return -1;
        if (right === data.default_model_id) return 1;
        return (models[left].label || left).localeCompare(models[right].label || right);
      });
      if (orderedModelIds.length === 0) {
        if (selectedModel) modelEl.value = selectedModel;
        return;
      }
      modelEl.innerHTML = "";
      for (const modelId of orderedModelIds) {
        const option = document.createElement("option");
        option.value = modelId;
        option.textContent = models[modelId].label || modelId;
        modelEl.appendChild(option);
      }
      modelEl.value = selectedModel;
    }

    async function refreshHealth() {
      try {
        const { data } = await fetchJson("/api/health", {}, 3000);
        setStatus("ready", "ready");
        precisionText.textContent = data.precision ? `· ${data.precision.toUpperCase()}` : "";
        syncModelOptions(data);
      } catch (_error) {
        setStatus("offline", "");
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const prompt = promptEl.value.trim();
      if (!prompt) return;

      promptEl.value = "";
      setStatus("queued", "busy");
      const modeLabel = agentModeEl.options[agentModeEl.selectedIndex].textContent;
      startActivity(modeLabel, "ジョブをキューに追加しています");
      addMessage("user", prompt);
      chatHistory.push({ role: "user", content: prompt });
      const requestId = makeRequestId();
      const jobThreadId = currentThreadId;
      const requestMessages = chatHistory.slice();
      const assistantNode = addMessage("assistant", "キューに追加しています...");
      const stepsNode = addStepPanel();
      renderSteps(stepsNode, clientInitialSteps(agentModeEl.value));
      trackActiveRequest(requestId);

      try {
        const res = await fetch("/api/chat/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            request_id: requestId,
            user_id: currentUserId,
            thread_id: jobThreadId,
            messages: requestMessages,
            model_id: modelEl.value,
            reasoning_effort: reasoningEl.value,
            agent_mode: agentModeEl.value
          })
        });
        if (!res.ok) {
          const data = await res.json();
          throw new Error(data.detail || "ジョブを開始できませんでした");
        }
        assistantNode.textContent = "順番待ちです。続けて送信できます。";
        sendEl.disabled = false;
        sendEl.textContent = "送信";
        setStatus("queued", "busy");
        finishActivity("ジョブを受け付けました");

        let lastStepUpdatedAt = 0;
        let lastJobUpdatedAt = 0;
        const poll = setInterval(async () => {
          try {
            const [stepsRes, jobRes] = await Promise.all([
              fetch(`/api/agent_steps/${encodeURIComponent(requestId)}`),
              fetch(`/api/chat/jobs/${encodeURIComponent(requestId)}`)
            ]);
            if (stepsRes.ok) {
              const stepData = await stepsRes.json();
              if ((stepData.updated_at || 0) !== lastStepUpdatedAt) {
                lastStepUpdatedAt = stepData.updated_at || 0;
                renderSteps(stepsNode, stepData.steps);
                const summary = activeStepSummary(stepData.steps);
                if (summary && !stepData.done) {
                  assistantNode.textContent = summary;
                }
              }
            }
            if (!jobRes.ok) return;
            const job = await jobRes.json();
            if ((job.updated_at || 0) === lastJobUpdatedAt && !job.done) return;
            lastJobUpdatedAt = job.updated_at || 0;
            if (!job.done) return;
            clearInterval(poll);
            untrackActiveRequest(requestId);
            if (job.error) {
              assistantNode.textContent = job.error === "Request cancelled" ? "キャンセルしました。" : `エラー: ${job.error}`;
              setStatus(job.error === "Request cancelled" ? "ready" : "error", job.error === "Request cancelled" ? "ready" : "");
              return;
            }
            const finalData = job.response;
            if (currentThreadId === jobThreadId) {
              assistantNode.textContent = finalData.reply || "";
              addMetrics(finalData);
              addSources(finalData.sources);
              chatHistory.push({ role: "assistant", content: finalData.reply || "" });
            }
            await loadThreads();
            setStatus("ready", "ready");
          } catch (error) {
            clearInterval(poll);
            untrackActiveRequest(requestId);
            assistantNode.textContent = `エラー: ${error.message}`;
            setStatus("error", "");
          }
        }, 1000);

        setStatus("ready", "ready");
      } catch (error) {
        untrackActiveRequest(requestId);
        assistantNode.textContent = `エラー: ${error.message}`;
        setStatus("error", "");
        finishActivity(`エラー: ${error.message}`);
      } finally {
        sendEl.disabled = false;
        sendEl.textContent = "送信";
        sendEl.classList.remove("cancel");
        promptEl.focus();
      }
    });

    cancelJobEl.addEventListener("click", cancelActiveRequest);

    modelEl.addEventListener("change", () => {
      addMessage("system", `${modelEl.value} に切り替えます。初回応答時にモデルをロードします。`);
      sendEl.disabled = true;
      setStatus("switching", "busy");
      startActivity("モデル切替", `${modelEl.value} を選択しています`);
      fetch("/api/switch_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelEl.value })
      }).catch((_error) => {});
      const selectedModel = modelEl.value;
      const waitForModel = setInterval(async () => {
        try {
          const res = await fetch("/api/health");
          const data = await res.json();
          if (data.default_model_id === selectedModel && (!data.model_loaded || data.active_model_id === selectedModel)) {
            clearInterval(waitForModel);
            sendEl.disabled = false;
            setStatus("ready", "ready");
            finishActivity("モデル切替が完了しました");
          }
        } catch (_error) {}
      }, 3000);
    });

    loginFormEl.addEventListener("submit", async (event) => {
      event.preventDefault();
      const displayName = loginNameEl.value.trim();
      if (!displayName) return;
      loginErrorEl.textContent = "";
      const loginButton = document.querySelector("#loginButton");
      loginButton.disabled = true;
      setStatus("login", "busy");
      try {
        await login(displayName);
      } catch (error) {
        loginErrorEl.textContent = error.message;
        setStatus("ready", "ready");
      } finally {
        loginButton.disabled = false;
      }
    });

    clearHistoryEl.addEventListener("click", async () => {
      if (isRunning) return;
      await fetch(`/api/history/${encodeURIComponent(currentUserId)}?thread_id=${encodeURIComponent(currentThreadId)}`, { method: "DELETE" });
      renderConversation([]);
      await loadThreads();
      finishActivity("履歴を削除しました");
    });

    newThreadEl.addEventListener("click", async (event) => {
      event.preventDefault();
      try {
        await createThread();
      } catch (error) {
        addMessage("system", `スレッドを作成できませんでした: ${error.message}`);
      }
    });

    deleteThreadEl.addEventListener("click", async () => {
      try {
        await deleteCurrentThread();
      } catch (error) {
        addMessage("system", `スレッドを削除できませんでした: ${error.message}`);
      }
    });

    logoutEl.addEventListener("click", async () => {
      if (isRunning) return;
      try {
        await fetch("/logout");
      } catch (_error) {}
      sessionStorage.removeItem("gaudiChatUserId");
      sessionStorage.removeItem("gaudiChatDisplayName");
      sessionStorage.removeItem("gaudiChatThreadId");
      currentUserId = "";
      currentDisplayName = "";
      currentThreadId = "default";
      renderConversation([]);
      showLogin();
    });

    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    if (currentUserId) {
      loadHistory(currentUserId).then(showChat).catch(showLogin);
    } else {
      showLogin();
    }
    refreshHealth();
  </script>
</body>
</html>
"""


def split_html_script() -> tuple[str, str]:
    script_open = "  <script>"
    script_close = "  </script>"
    start = HTML.index(script_open)
    end = HTML.index(script_close, start)
    script = HTML[start + len(script_open) : end].strip()
    shell = HTML[:start] + '  <script src="/app.js?v=9" defer></script>\n' + HTML[end + len(script_close) :]
    return shell, script


def render_html(
    user_id: str = "",
    display_name: str = "",
    messages: list | None = None,
    active_job: dict | None = None,
    thread_id: str = "default",
    thread_title: str = "メイン",
    threads: list | None = None,
) -> str:
    shell, _ = split_html_script()
    safe_user_id = html.escape(user_id, quote=True)
    safe_display_name = html.escape(display_name or user_id, quote=True)
    safe_thread_id = html.escape(thread_id or "default", quote=True)
    safe_thread_title = html.escape(thread_title or "メイン")
    model_options = "\n".join(
        f'          <option value="{html.escape(model_id, quote=True)}"'
        f'{" selected" if model_id == DEFAULT_MODEL_ID else ""}>'
        f'{html.escape(spec["label"])}</option>'
        for model_id, spec in supported_model_specs().items()
    )
    shell = shell.replace(
        '<body data-initial-user-id="" data-initial-display-name="">',
        f'<body data-initial-user-id="{safe_user_id}" data-initial-display-name="{safe_display_name}" '
        f'data-initial-thread-id="{safe_thread_id}">',
    )
    shell = re.sub(
        r'\s*<select id="model" name="model_id" aria-label="model">.*?</select>',
        f'\n              <select id="model" name="model_id" aria-label="model">\n{model_options}\n              </select>',
        shell,
        flags=re.DOTALL,
    )
    if user_id:
        rendered_messages = ['        <div class="message system">選択した Qwen モデルが Intel Gaudi HPU 上で応答します。</div>']
        for message in messages or []:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
            if role not in {"user", "assistant"}:
                continue
            rendered_messages.append(
                f'        <div class="message {role}">{html.escape(content)}</div>'
            )
        if active_job:
            step_lines = []
            for step in active_job.get("steps", []):
                status = html.escape(step.get("status", "pending"), quote=True)
                label = html.escape(step.get("label", "処理中"))
                detail = html.escape(step.get("detail", ""))
                detail_html = f'<div class="step-detail">{detail}</div>' if detail else ""
                step_lines.append(
                    f'          <div class="step {status}">'
                    f'<span class="step-marker"></span>'
                    f'<div><div class="step-title">{label}</div>{detail_html}</div>'
                    f'</div>'
                )
            if step_lines:
                title = "回答作成中" if not active_job.get("done") else "回答作成完了"
                rendered_messages.append(
                    '        <div class="steps" aria-live="polite">\n'
                    f'          <div class="step-title">{title}</div>\n'
                    + "\n".join(step_lines)
                    + "\n        </div>"
                )
        shell = re.sub(
            r'        <div class="message system">選択した Qwen モデルが Intel Gaudi HPU 上で応答します。</div>',
            "\n".join(rendered_messages),
            shell,
            count=1,
        )
        shell = shell.replace(
            '<section id="loginView" class="login-view">',
            '<section id="loginView" class="login-view hidden">',
        )
        shell = shell.replace('<div id="chatView" class="workspace hidden">', '<div id="chatView" class="workspace">')
        shell = shell.replace(
            '<strong id="currentUserName"></strong>',
            f'<strong id="currentUserName">{safe_display_name}</strong>',
        )
        shell = shell.replace(
            '<strong id="currentThreadTitle">メイン</strong>',
            f'<strong id="currentThreadTitle">{safe_thread_title}</strong>',
        )
        shell = shell.replace(
            '<input id="threadId" name="thread_id" type="hidden" value="default" />',
            f'<input id="threadId" name="thread_id" type="hidden" value="{safe_thread_id}" />',
        )
        if threads:
            rendered_threads = []
            for thread in threads:
                active_class = " active" if thread.thread_id == thread_id else ""
                running_label = "生成中 · " if getattr(thread, "running", False) else ""
                rendered_threads.append(
                    f'<a class="thread-item{active_class}" '
                    f'href="/?thread_id={html.escape(thread.thread_id, quote=True)}">'
                    f'{html.escape(running_label + thread.title)}</a>'
                )
            shell = shell.replace(
                '<div id="threadList" class="thread-list"></div>',
                f'<div id="threadList" class="thread-list">{"".join(rendered_threads)}</div>',
            )
        if active_job:
            state = "ready" if active_job.get("done") else "busy"
            mode = html.escape(active_job.get("mode", "回答作成中"))
            detail = html.escape(active_job.get("detail", "処理を進めています"))
            elapsed = ""
            started_at = active_job.get("started_at")
            if started_at and not active_job.get("done"):
                elapsed = f'{int(max(0, time.time() - float(started_at)))}s'
            shell = shell.replace(
                '<div id="activityBar" class="activitybar ready">',
                f'<div id="activityBar" class="activitybar {state}">',
            )
            shell = shell.replace(
                '<div id="activityMode" class="activity-mode">待機中</div>',
                f'<div id="activityMode" class="activity-mode">{mode}</div>',
            )
            shell = shell.replace(
                '<div id="activityDetail" class="activity-detail">メッセージを送信できます</div>',
                f'<div id="activityDetail" class="activity-detail">{detail}</div>',
            )
            shell = shell.replace(
                '<div id="activityElapsed" class="activity-elapsed"></div>',
                f'<div id="activityElapsed" class="activity-elapsed">{elapsed}</div>',
            )
    return shell


def app_javascript() -> str:
    _, script = split_html_script()
    return script + "\n"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    request_id: str | None = None
    user_id: str = "default"
    thread_id: str = "default"
    model_id: str = DEFAULT_MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    agent_mode: Literal["auto", "chat", "deep"] = "auto"
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.05, le=1.0)
    enable_thinking: bool | None = None


class SwitchModelRequest(BaseModel):
    model_id: str


class UserRequest(BaseModel):
    user_id: str | None = None
    display_name: str = Field(min_length=1, max_length=80)


class UserSummary(BaseModel):
    user_id: str
    display_name: str
    message_count: int
    updated_at: str | None = None


class HistoryResponse(BaseModel):
    user_id: str
    display_name: str
    messages: list[ChatMessage]


class ThreadRequest(BaseModel):
    user_id: str = "default"
    title: str | None = Field(default=None, max_length=120)


class ThreadUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    archived: bool | None = None


class ThreadSummary(BaseModel):
    thread_id: str
    title: str
    message_count: int
    last_mode: str = "auto"
    last_model_id: str | None = None
    updated_at: str | None = None
    running: bool = False


class ThreadHistoryResponse(BaseModel):
    user_id: str
    display_name: str
    thread_id: str
    title: str
    messages: list[ChatMessage]
    created_at: str | None = None
    updated_at: str | None = None


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchBundle(BaseModel):
    sources: list[SearchResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    label: str
    status: Literal["pending", "active", "done", "error"]
    detail: str = ""


class ChatResponse(BaseModel):
    reply: str
    precision: str
    reasoning_effort: str
    agent_mode: str
    resolved_mode: str = "chat"
    search_decision: str = ""
    sources: list[SearchResult] = Field(default_factory=list)
    effective_max_new_tokens: int
    enable_thinking: bool
    elapsed_sec: float
    ttft_sec: float
    tokens_per_sec: float
    generated_tokens: int


class AsyncJobResponse(BaseModel):
    request_id: str
    done: bool = False
    response: ChatResponse | None = None
    error: str | None = None
    updated_at: float


class RequestCancelled(Exception):
    pass


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _user_id(self, value: str | None, display_name: str | None = None) -> str:
        raw = (value or display_name or "default").strip().lower()
        user_id = re.sub(r"[^a-z0-9_.-]+", "-", raw).strip("-._")
        return user_id[:64] or "default"

    def _thread_id(self, value: str | None = None) -> str:
        raw = (value or str(uuid.uuid4())).strip().lower()
        thread_id = re.sub(r"[^a-z0-9_.-]+", "-", raw).strip("-._")
        return thread_id[:80] or str(uuid.uuid4())

    def _thread_title(self, title: str | None, messages: list | None = None) -> str:
        if title and title.strip():
            return title.strip()[:80]
        for message in messages or []:
            content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
            role = message.get("role", "") if isinstance(message, dict) else getattr(message, "role", "")
            if role == "user" and content:
                return content.strip().replace("\n", " ")[:40] or "新しいスレッド"
        return "無題"

    def _ensure_threads_unlocked(self, record: dict) -> dict:
        threads = record.get("threads")
        if not isinstance(threads, dict):
            legacy_messages = record.get("messages") if isinstance(record.get("messages"), list) else []
            created_at = record.get("created_at") or self._now()
            updated_at = record.get("updated_at") or created_at
            threads = {
                "default": {
                    "title": self._thread_title("メイン", legacy_messages),
                    "messages": legacy_messages,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "last_mode": "auto",
                    "last_model_id": None,
                    "archived": False,
                }
            }
            record["threads"] = threads
        if "active_thread_id" not in record or record["active_thread_id"] not in threads:
            record["active_thread_id"] = "default" if "default" in threads else next(iter(threads), "default")
        return threads

    def _load_unlocked(self) -> dict:
        if not self.path.exists():
            return {"users": {}}
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            return {"users": {}}
        if not isinstance(data, dict):
            return {"users": {}}
        users = data.get("users")
        if not isinstance(users, dict):
            data["users"] = {}
        return data

    def _save_unlocked(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        tmp_path.replace(self.path)

    def ensure_user(self, user_id: str | None = None, display_name: str | None = None) -> dict:
        normalized_user_id = self._user_id(user_id, display_name)
        label = (display_name or normalized_user_id).strip() or normalized_user_id
        with self.lock:
            data = self._load_unlocked()
            users = data["users"]
            record = users.get(normalized_user_id)
            if record is None:
                record = {
                    "display_name": label,
                    "messages": [],
                    "created_at": self._now(),
                    "updated_at": self._now(),
                }
                users[normalized_user_id] = record
            elif display_name:
                record["display_name"] = label
                record["updated_at"] = self._now()
            self._ensure_threads_unlocked(record)
            self._save_unlocked(data)
            return {"user_id": normalized_user_id, **record}

    def list_users(self) -> list[UserSummary]:
        self.ensure_user("default")
        with self.lock:
            data = self._load_unlocked()
            summaries = []
            for user_id, record in data["users"].items():
                messages = record.get("messages") if isinstance(record, dict) else []
                summaries.append(
                    UserSummary(
                        user_id=user_id,
                        display_name=record.get("display_name", user_id),
                        message_count=len(messages or []),
                        updated_at=record.get("updated_at"),
                    )
                )
        return sorted(summaries, key=lambda user: user.updated_at or "", reverse=True)

    def history(self, user_id: str) -> HistoryResponse:
        record = self.ensure_user(user_id)
        active_thread_id = record.get("active_thread_id", "default")
        return self.thread_history(user_id, active_thread_id)

    def list_threads(self, user_id: str) -> list[ThreadSummary]:
        normalized_user_id = self._user_id(user_id)
        self.ensure_user(normalized_user_id)
        with self.lock:
            data = self._load_unlocked()
            record = data["users"].get(normalized_user_id, {})
            threads = self._ensure_threads_unlocked(record)
            summaries = []
            for thread_id, thread in threads.items():
                if thread.get("archived"):
                    continue
                messages = thread.get("messages") if isinstance(thread.get("messages"), list) else []
                summaries.append(
                    ThreadSummary(
                        thread_id=thread_id,
                        title=thread.get("title") or self._thread_title(None, messages),
                        message_count=len(messages),
                        last_mode=thread.get("last_mode", "auto"),
                        last_model_id=thread.get("last_model_id"),
                        updated_at=thread.get("updated_at"),
                    )
                )
        return sorted(summaries, key=lambda thread: thread.updated_at or "", reverse=True)

    def create_thread(self, user_id: str, title: str | None = None) -> ThreadHistoryResponse:
        normalized_user_id = self._user_id(user_id)
        with self.lock:
            data = self._load_unlocked()
            users = data.setdefault("users", {})
            record = users.get(normalized_user_id)
            if record is None:
                record = {
                    "display_name": normalized_user_id,
                    "messages": [],
                    "created_at": self._now(),
                    "updated_at": self._now(),
                }
                users[normalized_user_id] = record
            threads = self._ensure_threads_unlocked(record)
            thread_id = self._thread_id()
            now = self._now()
            threads[thread_id] = {
                "title": self._thread_title(title),
                "messages": [],
                "created_at": now,
                "updated_at": now,
                "last_mode": "auto",
                "last_model_id": None,
                "archived": False,
            }
            record["active_thread_id"] = thread_id
            record["updated_at"] = now
            self._save_unlocked(data)
            return ThreadHistoryResponse(
                user_id=normalized_user_id,
                display_name=record.get("display_name", normalized_user_id),
                thread_id=thread_id,
                title=threads[thread_id]["title"],
                messages=[],
                created_at=now,
                updated_at=now,
            )

    def thread_history(self, user_id: str, thread_id: str = "default") -> ThreadHistoryResponse:
        normalized_user_id = self._user_id(user_id)
        record = self.ensure_user(normalized_user_id)
        normalized_thread_id = self._thread_id(thread_id)
        with self.lock:
            data = self._load_unlocked()
            record = data["users"].get(normalized_user_id, record)
            threads = self._ensure_threads_unlocked(record)
            if normalized_thread_id not in threads:
                normalized_thread_id = record.get("active_thread_id", "default")
            thread = threads.get(normalized_thread_id) or threads["default"]
            record["active_thread_id"] = normalized_thread_id
            self._save_unlocked(data)
        messages = []
        for message in thread.get("messages", []):
            try:
                messages.append(ChatMessage(**message))
            except Exception:
                continue
        return ThreadHistoryResponse(
            user_id=normalized_user_id,
            display_name=record.get("display_name", normalized_user_id),
            thread_id=normalized_thread_id,
            title=thread.get("title") or self._thread_title(None, messages),
            messages=messages,
            created_at=thread.get("created_at"),
            updated_at=thread.get("updated_at"),
        )

    def replace_history(
        self,
        user_id: str,
        messages: list[ChatMessage],
        thread_id: str = "default",
        last_mode: str | None = None,
        last_model_id: str | None = None,
    ) -> None:
        normalized_user_id = self._user_id(user_id)
        normalized_thread_id = self._thread_id(thread_id)
        with self.lock:
            data = self._load_unlocked()
            users = data["users"]
            record = users.get(normalized_user_id)
            if record is None:
                record = {
                    "display_name": normalized_user_id,
                    "messages": [],
                    "created_at": self._now(),
                }
                users[normalized_user_id] = record
            threads = self._ensure_threads_unlocked(record)
            if normalized_thread_id not in threads:
                now = self._now()
                threads[normalized_thread_id] = {
                    "title": self._thread_title(None, messages),
                    "messages": [],
                    "created_at": now,
                    "updated_at": now,
                    "last_mode": "auto",
                    "last_model_id": None,
                    "archived": False,
                }
            serialized = [message.model_dump() for message in messages]
            now = self._now()
            thread = threads[normalized_thread_id]
            thread["messages"] = serialized
            if not thread.get("title") or thread.get("title") in {"新しいスレッド", "無題"}:
                thread["title"] = self._thread_title(None, serialized)
            thread["updated_at"] = now
            if last_mode:
                thread["last_mode"] = last_mode
            if last_model_id:
                thread["last_model_id"] = last_model_id
            record["active_thread_id"] = normalized_thread_id
            record["messages"] = serialized if normalized_thread_id == "default" else record.get("messages", [])
            record["updated_at"] = now
            self._save_unlocked(data)

    def append_message(
        self,
        user_id: str,
        thread_id: str,
        message: ChatMessage,
        last_mode: str | None = None,
        last_model_id: str | None = None,
    ) -> None:
        normalized_user_id = self._user_id(user_id)
        normalized_thread_id = self._thread_id(thread_id)
        with self.lock:
            data = self._load_unlocked()
            users = data.setdefault("users", {})
            record = users.get(normalized_user_id)
            if record is None:
                record = {
                    "display_name": normalized_user_id,
                    "messages": [],
                    "created_at": self._now(),
                }
                users[normalized_user_id] = record
            threads = self._ensure_threads_unlocked(record)
            if normalized_thread_id not in threads:
                now = self._now()
                threads[normalized_thread_id] = {
                    "title": self._thread_title(None, [message]),
                    "messages": [],
                    "created_at": now,
                    "updated_at": now,
                    "last_mode": "auto",
                    "last_model_id": None,
                    "archived": False,
                }
            thread = threads[normalized_thread_id]
            messages = thread.get("messages")
            if not isinstance(messages, list):
                messages = []
                thread["messages"] = messages
            messages.append(message.model_dump())
            if not thread.get("title") or thread.get("title") in {"新しいスレッド", "無題"}:
                thread["title"] = self._thread_title(None, messages)
            now = self._now()
            thread["updated_at"] = now
            if last_mode:
                thread["last_mode"] = last_mode
            if last_model_id:
                thread["last_model_id"] = last_model_id
            record["active_thread_id"] = normalized_thread_id
            if normalized_thread_id == "default":
                record["messages"] = messages
            record["updated_at"] = now
            self._save_unlocked(data)

    def update_thread(
        self,
        user_id: str,
        thread_id: str,
        title: str | None = None,
        archived: bool | None = None,
    ) -> ThreadHistoryResponse:
        normalized_user_id = self._user_id(user_id)
        normalized_thread_id = self._thread_id(thread_id)
        with self.lock:
            data = self._load_unlocked()
            record = data["users"].get(normalized_user_id)
            if record is None:
                raise KeyError("user not found")
            threads = self._ensure_threads_unlocked(record)
            if normalized_thread_id not in threads:
                raise KeyError("thread not found")
            thread = threads[normalized_thread_id]
            if title is not None and title.strip():
                thread["title"] = title.strip()[:80]
            if archived is not None:
                thread["archived"] = archived
            thread["updated_at"] = self._now()
            record["updated_at"] = thread["updated_at"]
            self._save_unlocked(data)
        return self.thread_history(normalized_user_id, normalized_thread_id)

    def clear_history(self, user_id: str, thread_id: str = "default") -> None:
        normalized_user_id = self._user_id(user_id)
        normalized_thread_id = self._thread_id(thread_id)
        with self.lock:
            data = self._load_unlocked()
            record = data["users"].get(normalized_user_id)
            if record is None:
                return
            threads = self._ensure_threads_unlocked(record)
            thread = threads.get(normalized_thread_id)
            if thread is None:
                return
            thread["messages"] = []
            thread["updated_at"] = self._now()
            if normalized_thread_id == "default":
                record["messages"] = []
            record["updated_at"] = thread["updated_at"]
            self._save_unlocked(data)


class CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_event: threading.Event | None, prompt_len: int | None = None) -> None:
        self.cancel_event = cancel_event
        self.prompt_len = prompt_len
        self.start_time: float | None = None
        self.first_token_time: float | None = None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.prompt_len is not None and input_ids.shape[-1] > self.prompt_len:
            if self.first_token_time is None:
                self.first_token_time = time.perf_counter()
        return bool(self.cancel_event and self.cancel_event.is_set())

    @property
    def ttft(self) -> float | None:
        if self.start_time is None or self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time


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


def parse_duckduckgo_results(body: str, limit: int) -> list[SearchResult]:
    pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?\[(?P<title>[^\]\n]+)\]\((?P<url>https?://[^)]+)\)"
        r"(?P<snippet>[^\n]*)",
        re.DOTALL,
    )
    results = []
    seen = set()
    for match in pattern.finditer(body):
        title = strip_tags(match.group("title"))
        result_url = normalize_duckduckgo_url(html.unescape(match.group("url")))
        snippet = strip_tags(match.group("snippet"))
        parsed = urlparse(result_url)
        if not title or not parsed.netloc or "duckduckgo.com" in parsed.netloc:
            continue
        if result_url in seen:
            continue
        seen.add(result_url)
        results.append(SearchResult(title=title, url=result_url, snippet=snippet))
        if len(results) >= limit:
            break
    return results


def parse_bing_results(body: str, limit: int) -> list[SearchResult]:
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


def web_search(query: str, limit: int = 5) -> list[SearchResult]:
    engine = SEARCH_ENGINE.lower()
    if engine == "bing":
        url = f"https://r.jina.ai/http://www.bing.com/search?q={quote_plus(query)}"
    else:
        url = f"https://r.jina.ai/http://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=SEARCH_TIMEOUT_SEC,
    )
    response.raise_for_status()
    body = response.text
    if engine == "bing":
        return parse_bing_results(body, limit)
    return parse_duckduckgo_results(body, limit)


def search_queries(prompt: str, mode: str) -> list[str]:
    if mode == "web":
        return [prompt]
    return [
        prompt,
        f"{prompt} 最新",
        f"{prompt} 背景 解説",
    ]


def collect_sources(prompt: str, mode: str, cancel_event: threading.Event | None = None) -> SearchBundle:
    if mode == "chat":
        return SearchBundle()
    sources = []
    warnings = []
    seen = set()
    per_query = 4 if mode == "deep" else 5
    max_sources = 8 if mode == "deep" else 5
    for query in search_queries(prompt, mode):
        if cancel_event and cancel_event.is_set():
            raise RequestCancelled()
        try:
            results = web_search(query, limit=per_query)
        except requests.RequestException as exc:
            warnings.append(f"{query}: {exc}")
            continue
        for result in results:
            if cancel_event and cancel_event.is_set():
                raise RequestCancelled()
            if result.url in seen:
                continue
            seen.add(result.url)
            sources.append(result)
            if len(sources) >= max_sources:
                return SearchBundle(sources=sources, warnings=warnings)
    return SearchBundle(sources=sources, warnings=warnings)


AUTO_SEARCH_PATTERNS = [
    r"最新",
    r"現在",
    r"今\b",
    r"今日",
    r"昨日",
    r"明日",
    r"今年",
    r"直近",
    r"速報",
    r"ニュース",
    r"障害",
    r"価格",
    r"株価",
    r"為替",
    r"天気",
    r"結果",
    r"日程",
    r"リリース",
    r"アップデート",
    r"検索",
    r"調べ",
    r"ソース",
    r"出典",
    r"根拠",
]
DEEP_SEARCH_PATTERNS = [
    r"詳しく",
    r"深く",
    r"網羅",
    r"比較",
    r"複数",
    r"背景",
    r"調査",
    r"まとめて",
]
NO_SEARCH_PATTERNS = [
    r"検索しない",
    r"調べない",
    r"外部情報なし",
]


def auto_search_decision(prompt: str, messages: list[ChatMessage] | None = None) -> tuple[str, str]:
    text = prompt.strip()
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in NO_SEARCH_PATTERNS):
        return "chat", "ユーザーが検索しないよう指定したため、モデルのみで回答"
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in DEEP_SEARCH_PATTERNS):
        return "deep", "複数観点の整理が必要なため Deep search を実行"
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in AUTO_SEARCH_PATTERNS):
        return "web", "最新性または出典確認が必要なため自動検索を実行"
    return "chat", "検索不要と判断し、モデルのみで回答"


def resolve_agent_mode(request: ChatRequest) -> tuple[str, str]:
    if request.agent_mode == "chat":
        return "chat", "Chat が選択されたため検索しません"
    if request.agent_mode == "deep":
        return "deep", "Deep search が選択されたため詳細検索を実行"
    return auto_search_decision(request.messages[-1].content, request.messages)


def request_with_sources(request: ChatRequest, sources: list[SearchResult]) -> ChatRequest:
    if not sources:
        return request
    messages = list(request.messages)
    last = messages[-1]
    source_lines = "\n".join(
        f"[{index}] {source.title}\nURL: {source.url}\n概要: {source.snippet}"
        for index, source in enumerate(sources, start=1)
    )
    resolved_mode, _ = resolve_agent_mode(request)
    mode_name = AGENT_MODES[resolved_mode]["label"]
    enriched = (
        f"{last.content}\n\n"
        f"{mode_name} の検索結果:\n{source_lines}\n\n"
        "上の検索結果を根拠として使い、必要なら [1] のように番号で出典を示して日本語で答えてください。"
        "途中で切らず、長くなりそうな場合は要点を絞って最後まで完結させてください。"
    )
    messages[-1] = ChatMessage(role=last.role, content=enriched)
    return request.model_copy(update={"messages": messages})


def enable_optimum_habana() -> bool:
    global OPTIMUM_HABANA_ENABLED
    if OPTIMUM_HABANA_ENABLED is not None:
        return OPTIMUM_HABANA_ENABLED

    python_bin_dir = os.path.dirname(sys.executable)
    os.environ["PATH"] = f"{python_bin_dir}:{os.environ.get('PATH', '')}"
    try:
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    except Exception as error:
        print(f"Optimum Habana unavailable: {error}", flush=True)
        OPTIMUM_HABANA_ENABLED = False
        return False

    adapt_transformers_to_gaudi()
    print("Optimum Habana Gaudi patches enabled.", flush=True)
    OPTIMUM_HABANA_ENABLED = True
    return True


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
        if not is_model_supported(model_id):
            required_type = MODEL_REQUIRED_TYPES[model_id]
            raise ValueError(
                f"{model_id} requires Transformers support for '{required_type}'. "
                f"This environment provides transformers=={package_version('transformers')}; "
                f"use {COMPAT_DEFAULT_MODEL_ID} or run with /home/test1/habanalabs-venv/bin/python."
            )
        if self.is_loaded and self.active_model_id == model_id:
            return

        if not torch.hpu.is_available():
            raise RuntimeError("HPU is not available. Check Habana driver/runtime setup.")

        self.unload()
        htcore.hpu_inference_set_env()
        optimum_enabled = enable_optimum_habana()
        spec = MODEL_SPECS[model_id]
        execution_model_id = resolve_execution_model_id(model_id)
        started = time.time()
        print(
            f"Loading model: requested={model_id}, execution={execution_model_id}, "
            f"kind={spec['kind']}, precision={spec['precision']}",
            flush=True,
        )
        pretrained_source, pretrained_kwargs = resolve_pretrained_source(execution_model_id)
        if spec["kind"] == "image_text":
            tokenizer = AutoProcessor.from_pretrained(pretrained_source, **pretrained_kwargs)
            model_cls = AutoModelForImageTextToText
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_source, **pretrained_kwargs)
            model_cls = AutoModelForCausalLM

        if is_fp8_model(execution_model_id):
            model_kwargs = {}
        elif spec["kind"] == "image_text":
            model_kwargs = {"dtype": torch.bfloat16}
        else:
            model_kwargs = {"torch_dtype": torch.bfloat16}
        self.model = model_cls.from_pretrained(
            pretrained_source,
            **model_kwargs,
            **pretrained_kwargs,
            low_cpu_mem_usage=True,
            device_map={"": "hpu"},
        )
        self.model.eval()
        self.tokenizer = tokenizer
        self.active_model_id = model_id
        self.model_kind = spec["kind"]
        htcore.mark_step()
        torch.hpu.synchronize()
        self.precision = (
            f"bf16 fallback from {model_id}" if execution_model_id != model_id else spec["precision"]
        )
        self.loaded_at = time.time()
        print(
            f"Model load complete: requested={model_id}, active={self.active_model_id}, "
            f"elapsed={self.loaded_at - started:.1f}s, precision={self.precision}",
            flush=True,
        )
        if not optimum_enabled:
            print("Continuing with Transformers + habana_frameworks fallback.", flush=True)

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

    def generate(
        self,
        request: ChatRequest,
        sources: list[SearchResult] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ChatResponse:
        sources = sources or []
        with self.lock:
            if cancel_event and cancel_event.is_set():
                raise RequestCancelled()
            self.load(request.model_id)
            assert self.tokenizer is not None
            assert self.model is not None
            if cancel_event and cancel_event.is_set():
                raise RequestCancelled()

            text = self.render_prompt(request)
            inputs = self.tokenize(text)
            if USE_INT32_INPUTS:
                for key in ("input_ids", "attention_mask"):
                    if key in inputs:
                        inputs[key] = inputs[key].to(dtype=torch.int32)
            inputs = {key: value.to("hpu") for key, value in inputs.items()}

            prompt_len = inputs["input_ids"].shape[-1]
            do_sample = request.temperature > 0
            stopping_criteria = CancelStoppingCriteria(cancel_event, prompt_len)
            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.max_new_tokens(request),
                "do_sample": do_sample,
                "pad_token_id": self.eos_token_id(),
                "use_cache": True,
                "stopping_criteria": StoppingCriteriaList([stopping_criteria]),
            }
            if do_sample:
                generation_kwargs["temperature"] = request.temperature
                generation_kwargs["top_p"] = request.top_p

            torch.hpu.synchronize()
            started = time.perf_counter()
            stopping_criteria.start()
            with torch.inference_mode():
                output_ids = self.model.generate(**generation_kwargs)
                htcore.mark_step()
                torch.hpu.synchronize()

            if cancel_event and cancel_event.is_set():
                raise RequestCancelled()

            finished = time.perf_counter()
            generated_ids = output_ids[:, prompt_len:]
            generated_tokens = int(generated_ids.shape[-1])
            reply = self.clean_reply(self.decode(generated_ids))
            if not reply:
                reply = "High の推論でトークン上限に達しました。Medium に下げるか、上限を増やしてください。"

            elapsed_sec = finished - started
            ttft_sec = stopping_criteria.ttft or elapsed_sec
            decode_sec = max(elapsed_sec - (stopping_criteria.ttft or 0.0), 1e-9)
            tps_tokens = max(generated_tokens - 1, 0) if stopping_criteria.ttft is not None else generated_tokens
            tokens_per_sec = tps_tokens / decode_sec
            return ChatResponse(
                reply=reply,
                precision=self.precision,
                reasoning_effort=request.reasoning_effort,
                agent_mode=request.agent_mode,
                sources=sources,
                effective_max_new_tokens=self.max_new_tokens(request),
                enable_thinking=self.enable_thinking(request),
                elapsed_sec=elapsed_sec,
                ttft_sec=ttft_sec,
                tokens_per_sec=tokens_per_sec,
                generated_tokens=generated_tokens,
            )

    def stream_generate(
        self,
        request: ChatRequest,
        sources: list[SearchResult] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        sources = sources or []
        with self.lock:
            if cancel_event and cancel_event.is_set():
                raise RequestCancelled()
            self.load(request.model_id)
            assert self.tokenizer is not None
            assert self.model is not None
            if cancel_event and cancel_event.is_set():
                raise RequestCancelled()

            text = self.render_prompt(request)
            inputs = self.tokenize(text)
            if USE_INT32_INPUTS:
                for key in ("input_ids", "attention_mask"):
                    if key in inputs:
                        inputs[key] = inputs[key].to(dtype=torch.int32)
            inputs = {key: value.to("hpu") for key, value in inputs.items()}

            prompt_len = inputs["input_ids"].shape[-1]
            do_sample = request.temperature > 0
            stopping_criteria = CancelStoppingCriteria(cancel_event, prompt_len)
            streamer = TextIteratorStreamer(
                self.tokenizer.tokenizer if hasattr(self.tokenizer, "tokenizer") else self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=1.0,
            )
            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.max_new_tokens(request),
                "do_sample": do_sample,
                "pad_token_id": self.eos_token_id(),
                "use_cache": True,
                "stopping_criteria": StoppingCriteriaList([stopping_criteria]),
                "streamer": streamer,
            }
            if do_sample:
                generation_kwargs["temperature"] = request.temperature
                generation_kwargs["top_p"] = request.top_p

            generated_output = {}
            generation_error = {}

            def run_generate() -> None:
                try:
                    with torch.inference_mode():
                        generated_output["output_ids"] = self.model.generate(**generation_kwargs)
                        htcore.mark_step()
                        torch.hpu.synchronize()
                except Exception as exc:
                    generation_error["error"] = exc

            torch.hpu.synchronize()
            started = time.perf_counter()
            stopping_criteria.start()
            worker = threading.Thread(target=run_generate)
            worker.start()

            chunks = []
            iterator = iter(streamer)
            while True:
                if cancel_event and cancel_event.is_set():
                    worker.join(timeout=1.0)
                    raise RequestCancelled()
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                except queue.Empty:
                    if generation_error:
                        raise generation_error["error"]
                    if not worker.is_alive():
                        break
                    continue
                if chunk:
                    chunks.append(chunk)
                    yield {"type": "delta", "text": chunk}

            worker.join()
            if generation_error:
                raise generation_error["error"]
            if cancel_event and cancel_event.is_set():
                raise RequestCancelled()

            finished = time.perf_counter()
            output_ids = generated_output["output_ids"]
            generated_ids = output_ids[:, prompt_len:]
            generated_tokens = int(generated_ids.shape[-1])
            reply = self.clean_reply("".join(chunks))
            if not reply:
                reply = self.clean_reply(self.decode(generated_ids))
            if not reply:
                reply = "High の推論でトークン上限に達しました。Medium に下げるか、上限を増やしてください。"

            elapsed_sec = finished - started
            ttft_sec = stopping_criteria.ttft or elapsed_sec
            decode_sec = max(elapsed_sec - (stopping_criteria.ttft or 0.0), 1e-9)
            tps_tokens = max(generated_tokens - 1, 0) if stopping_criteria.ttft is not None else generated_tokens
            tokens_per_sec = tps_tokens / decode_sec
            response = ChatResponse(
                reply=reply,
                precision=self.precision,
                reasoning_effort=request.reasoning_effort,
                agent_mode=request.agent_mode,
                sources=sources,
                effective_max_new_tokens=self.max_new_tokens(request),
                enable_thinking=self.enable_thinking(request),
                elapsed_sec=elapsed_sec,
                ttft_sec=ttft_sec,
                tokens_per_sec=tokens_per_sec,
                generated_tokens=generated_tokens,
            )
            yield {"type": "final", "data": response.model_dump()}

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
        if request.max_new_tokens is not None:
            return request.max_new_tokens
        mode_tokens = AGENT_MODE_MIN_TOKENS.get(request.agent_mode, 0)
        return max(int(preset_tokens), mode_tokens)

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
    history_store = HistoryStore(HISTORY_PATH)
    progress_lock = threading.Lock()
    agent_progress: dict[str, dict] = {}
    cancel_events: dict[str, threading.Event] = {}
    progress_threads: dict[str, dict[str, str]] = {}
    async_jobs: dict[str, dict] = {}
    fallback_lock = threading.Lock()
    fallback_jobs: dict[str, dict] = {}

    def request_key(request_id: str | None) -> str:
        return request_id or str(uuid.uuid4())

    def set_steps(request_id: str, steps: list[AgentStep], done: bool = False) -> None:
        with progress_lock:
            agent_progress[request_id] = {
                "steps": [step.model_dump() for step in steps],
                "done": done,
                "updated_at": time.time(),
            }

    def set_progress_thread(request_id: str, user_id: str, thread_id: str) -> None:
        with progress_lock:
            progress_threads[request_id] = {"user_id": user_id, "thread_id": thread_id}

    def set_async_job(request_id: str, **updates) -> None:
        with progress_lock:
            job = async_jobs.get(request_id)
            if not job:
                return
            job.update(updates)
            job["updated_at"] = time.time()

    def cancel_event_for(request_id: str) -> threading.Event:
        with progress_lock:
            event = cancel_events.get(request_id)
            if event is None:
                event = threading.Event()
                cancel_events[request_id] = event
            return event

    def update_step(
        request_id: str,
        steps: list[AgentStep],
        index: int,
        status: Literal["pending", "active", "done", "error"],
        detail: str = "",
        done: bool = False,
    ) -> None:
        steps[index].status = status
        steps[index].detail = detail
        set_steps(request_id, steps, done=done)

    def initial_steps(mode: str) -> list[AgentStep]:
        steps = [AgentStep(label="リクエスト受付", status="done", detail=AGENT_MODES[mode]["label"])]
        if mode == "auto":
            steps.append(AgentStep(label="検索要否判定", status="pending"))
        if mode in {"web", "deep"}:
            steps.append(AgentStep(label="自動検索" if mode == "web" else "Deep search", status="pending"))
            steps.append(AgentStep(label="検索結果の整理", status="pending"))
        steps.append(AgentStep(label="プロンプト作成", status="pending"))
        steps.append(AgentStep(label="モデル生成", status="pending"))
        steps.append(AgentStep(label="応答完了", status="pending"))
        return steps

    def insert_search_steps(steps: list[AgentStep], resolved_mode: str) -> int:
        if resolved_mode not in {"web", "deep"}:
            return 0
        insert_at = 2 if len(steps) > 1 and steps[1].label == "検索要否判定" else 1
        if not any(step.label in {"自動検索", "Deep search", "Web検索"} for step in steps):
            steps.insert(
                insert_at,
                AgentStep(label="自動検索" if resolved_mode == "web" else "Deep search", status="pending"),
            )
            steps.insert(insert_at + 1, AgentStep(label="検索結果の整理", status="pending"))
        return insert_at

    def set_fallback_job(job_id: str, **updates) -> None:
        with fallback_lock:
            job = fallback_jobs.get(job_id)
            if not job:
                return
            job.update(updates)
            job["updated_at"] = time.time()

    def set_fallback_step(
        job_id: str,
        steps: list[AgentStep],
        index: int,
        status: Literal["pending", "active", "done", "error"],
        detail: str = "",
        done: bool = False,
    ) -> None:
        steps[index].status = status
        steps[index].detail = detail
        set_fallback_job(
            job_id,
            steps=[step.model_dump() for step in steps],
            detail=detail or steps[index].label,
            done=done,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Starting Gaudi Qwen chat server", flush=True)
        print(f"torch={package_version('torch')}", flush=True)
        print(f"habana-torch-plugin={package_version('habana-torch-plugin')}", flush=True)
        print(f"optimum-habana={package_version('optimum-habana')}", flush=True)
        print(f"transformers={package_version('transformers')}", flush=True)
        print(f"default_model_id={engine.default_model_id}", flush=True)
        yield

    app = FastAPI(title="Gaudi Qwen Chat", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        user_id = request.cookies.get("gaudi_chat_user_id", "")
        display_name = unquote(request.cookies.get("gaudi_chat_display_name", user_id))
        job_id = request.query_params.get("job_id", "")
        thread_id = request.query_params.get("thread_id", "default") or "default"
        thread_title = "メイン"
        threads = []
        active_job = None
        messages = []
        if user_id:
            try:
                history = history_store.thread_history(user_id, thread_id)
                display_name = history.display_name
                messages = history.messages
                thread_id = history.thread_id
                thread_title = history.title
                threads = history_store.list_threads(user_id)
            except Exception:
                messages = []
            if job_id:
                with fallback_lock:
                    record = fallback_jobs.get(job_id)
                    if record and record.get("user_id") == user_id:
                        active_job = dict(record)
        return HTMLResponse(
            render_html(user_id, display_name, messages, active_job, thread_id, thread_title, threads),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.get("/app.js")
    def app_js() -> Response:
        return Response(
            app_javascript(),
            media_type="application/javascript; charset=utf-8",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.get("/login")
    def login_fallback(display_name: str = "") -> RedirectResponse:
        if not display_name.strip():
            return RedirectResponse("/", status_code=303)
        record = history_store.ensure_user(display_name=display_name)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie("gaudi_chat_user_id", record["user_id"], samesite="lax")
        response.set_cookie(
            "gaudi_chat_display_name",
            quote_plus(record.get("display_name", record["user_id"])),
            samesite="lax",
        )
        return response

    @app.get("/logout")
    def logout_fallback() -> RedirectResponse:
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie("gaudi_chat_user_id")
        response.delete_cookie("gaudi_chat_display_name")
        return response

    @app.get("/threads/new")
    def new_thread_fallback(request: Request) -> RedirectResponse:
        user_id = request.cookies.get("gaudi_chat_user_id", "")
        if not user_id:
            return RedirectResponse("/", status_code=303)
        thread = history_store.create_thread(user_id)
        return RedirectResponse(f"/?thread_id={thread.thread_id}", status_code=303)

    def run_fallback_chat(job_id: str, chat_request: ChatRequest) -> None:
        steps = initial_steps(chat_request.agent_mode)
        set_fallback_job(
            job_id,
            steps=[step.model_dump() for step in steps],
            mode=AGENT_MODES[chat_request.agent_mode]["label"],
            detail="回答作成を開始しています",
        )
        try:
            print(
                f"Fallback chat start: job_id={job_id}, user={chat_request.user_id}, "
                f"mode={chat_request.agent_mode}, model={chat_request.model_id}",
                flush=True,
            )
            if engine.is_loaded and engine.active_model_id != chat_request.model_id:
                raise RuntimeError(
                    "Select the model in the UI first. The server restarts to switch models cleanly."
                )

            cursor = 1
            sources: list[SearchResult] = []
            resolved_mode, search_decision = resolve_agent_mode(chat_request)
            if chat_request.agent_mode == "auto":
                set_fallback_step(job_id, steps, cursor, "active", "検索が必要か判定しています")
                set_fallback_step(job_id, steps, cursor, "done", search_decision)
                cursor += 1
            if resolved_mode in {"web", "deep"}:
                search_index = insert_search_steps(steps, resolved_mode)
                if search_index:
                    cursor = search_index
                    set_fallback_job(job_id, steps=[step.model_dump() for step in steps])
                set_fallback_step(job_id, steps, cursor, "active", "検索クエリを実行しています")
                search_bundle = collect_sources(
                    chat_request.messages[-1].content,
                    resolved_mode,
                )
                sources = search_bundle.sources
                detail = f"{len(sources)} 件の候補を取得"
                if search_bundle.warnings:
                    detail = f"{len(sources)} 件取得。一部検索失敗: {len(search_bundle.warnings)} 件"
                set_fallback_step(job_id, steps, cursor, "done", detail)
                cursor += 1

                set_fallback_step(job_id, steps, cursor, "active", "検索結果を回答用に整理しています")
                if sources:
                    set_fallback_step(job_id, steps, cursor, "done", f"{min(len(sources), 8)} 件をコンテキスト化")
                else:
                    set_fallback_step(job_id, steps, cursor, "done", "検索結果なし。通常プロンプトで続行")
                cursor += 1

            set_fallback_step(job_id, steps, cursor, "active", "会話履歴と条件をまとめています")
            enriched_request = request_with_sources(chat_request, sources)
            if resolved_mode == "deep" and enriched_request.max_new_tokens is None:
                enriched_request = enriched_request.model_copy(update={"max_new_tokens": 4096})
            set_fallback_step(job_id, steps, cursor, "done", "生成用プロンプトを作成")
            cursor += 1

            detail = "モデルをロードして生成しています" if not engine.is_loaded else "HPU 上のモデルで生成しています"
            set_fallback_step(job_id, steps, cursor, "active", detail)
            response = engine.generate(enriched_request, sources=sources)
            set_fallback_step(job_id, steps, cursor, "done", "生成が完了しました")
            cursor += 1

            history = history_store.thread_history(chat_request.user_id, chat_request.thread_id)
            saved_messages = list(history.messages)
            saved_messages.append(ChatMessage(role="assistant", content=response.reply))
            history_store.replace_history(
                chat_request.user_id,
                saved_messages,
                chat_request.thread_id,
                last_mode=chat_request.agent_mode,
                last_model_id=chat_request.model_id,
            )
            set_fallback_step(job_id, steps, cursor, "done", "チャット画面へ回答を保存しました", done=True)
            set_fallback_job(job_id, mode="完了", detail="回答を表示しました", done=True)
            print(f"Fallback chat complete: job_id={job_id}, user={chat_request.user_id}", flush=True)
        except Exception as exc:
            print(f"Fallback chat error: job_id={job_id}, user={chat_request.user_id}, error={exc}", flush=True)
            for step in steps:
                if step.status == "active":
                    step.status = "error"
                    step.detail = str(exc)
            steps[-1].status = "error"
            steps[-1].detail = str(exc)
            try:
                history = history_store.thread_history(chat_request.user_id, chat_request.thread_id)
                saved_messages = list(history.messages)
                saved_messages.append(ChatMessage(role="assistant", content=f"エラー: {exc}"))
                history_store.replace_history(chat_request.user_id, saved_messages, chat_request.thread_id)
            finally:
                set_fallback_job(
                    job_id,
                    mode="エラー",
                    detail=str(exc),
                    steps=[step.model_dump() for step in steps],
                    done=True,
                )

    @app.post("/chat/send")
    async def chat_send_fallback(request: Request) -> RedirectResponse:
        user_id = request.cookies.get("gaudi_chat_user_id", "")
        if not user_id:
            return RedirectResponse("/", status_code=303)

        form = parse_qs((await request.body()).decode("utf-8", errors="ignore"))

        def field(name: str, default: str = "") -> str:
            values = form.get(name)
            return values[0].strip() if values else default

        prompt = field("prompt")
        if not prompt:
            return RedirectResponse("/", status_code=303)
        thread_id = field("thread_id", "default")

        model_id = field("model_id", engine.default_model_id)
        if model_id not in supported_model_specs():
            model_id = engine.default_model_id
        reasoning_effort = field("reasoning_effort", "medium")
        if reasoning_effort not in REASONING_PRESETS:
            reasoning_effort = "medium"
        agent_mode = field("agent_mode", "auto" if AUTO_SEARCH_DEFAULT else "chat")
        if agent_mode not in USER_AGENT_MODES:
            agent_mode = "auto" if AUTO_SEARCH_DEFAULT else "chat"

        history = history_store.thread_history(user_id, thread_id)
        messages = list(history.messages)
        messages.append(ChatMessage(role="user", content=prompt))
        history_store.replace_history(user_id, messages, thread_id, last_mode=agent_mode, last_model_id=model_id)
        chat_request = ChatRequest(
            user_id=user_id,
            thread_id=thread_id,
            model_id=model_id,
            messages=messages,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            agent_mode=agent_mode,  # type: ignore[arg-type]
        )
        job_id = str(uuid.uuid4())
        steps = initial_steps(agent_mode)
        with fallback_lock:
            fallback_jobs[job_id] = {
                "user_id": user_id,
                "thread_id": thread_id,
                "mode": AGENT_MODES[agent_mode]["label"],
                "detail": "回答作成を開始しています",
                "steps": [step.model_dump() for step in steps],
                "done": False,
                "started_at": time.time(),
                "updated_at": time.time(),
            }
        worker = threading.Thread(target=run_fallback_chat, args=(job_id, chat_request), daemon=True)
        worker.start()
        return RedirectResponse(f"/?thread_id={thread_id}&job_id={job_id}", status_code=303)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "default_model_id": engine.default_model_id,
            "models": supported_model_specs(),
            "reasoning_presets": REASONING_PRESETS,
            "agent_modes": USER_AGENT_MODES,
            "internal_agent_modes": AGENT_MODES,
            "auto_search_default": AUTO_SEARCH_DEFAULT,
            "search_engine": SEARCH_ENGINE.lower(),
            "search_timeout_sec": SEARCH_TIMEOUT_SEC,
            "hpu_execution_mode": HPU_EXECUTION_MODE,
            "use_int32_inputs": USE_INT32_INPUTS,
            "model_placement": MODEL_PLACEMENT,
            "active_model_id": engine.active_model_id,
            "model_loaded": engine.is_loaded,
            "precision": engine.precision,
            "fp8_enabled": engine.precision.startswith("fp8"),
            "hpu_available": None,
            "hpu_devices": None,
            "hpu_current_device": None,
            "loaded_at": engine.loaded_at,
        }

    @app.get("/api/agent_steps/{request_id}")
    def agent_steps(request_id: str) -> dict:
        with progress_lock:
            record = agent_progress.get(request_id)
        if record is None:
            return {"steps": [], "done": False}
        return {"steps": record["steps"], "done": record["done"], "updated_at": record["updated_at"]}

    @app.post("/api/users")
    def save_user(request: UserRequest, response: Response) -> dict:
        record = history_store.ensure_user(request.user_id, request.display_name)
        response.set_cookie("gaudi_chat_user_id", record["user_id"], samesite="lax")
        response.set_cookie(
            "gaudi_chat_display_name",
            quote_plus(record.get("display_name", record["user_id"])),
            samesite="lax",
        )
        return {
            "user_id": record["user_id"],
            "display_name": record.get("display_name", record["user_id"]),
        }

    @app.get("/api/threads/{user_id}", response_model=list[ThreadSummary])
    def user_threads(user_id: str) -> list[ThreadSummary]:
        summaries = history_store.list_threads(user_id)
        running_threads = set()
        with progress_lock:
            for request_id, owner in progress_threads.items():
                record = agent_progress.get(request_id)
                if owner.get("user_id") == user_id and record and not record.get("done"):
                    running_threads.add(owner.get("thread_id", "default"))
        with fallback_lock:
            for job in fallback_jobs.values():
                if job.get("user_id") == user_id and not job.get("done"):
                    running_threads.add(job.get("thread_id", "default"))
        return [
            summary.model_copy(update={"running": summary.thread_id in running_threads})
            for summary in summaries
        ]

    @app.post("/api/threads", response_model=ThreadHistoryResponse)
    def create_thread(request: ThreadRequest) -> ThreadHistoryResponse:
        return history_store.create_thread(request.user_id, request.title)

    @app.get("/api/threads/{user_id}/{thread_id}", response_model=ThreadHistoryResponse)
    def thread_history(user_id: str, thread_id: str) -> ThreadHistoryResponse:
        return history_store.thread_history(user_id, thread_id)

    @app.patch("/api/threads/{user_id}/{thread_id}", response_model=ThreadHistoryResponse)
    def update_thread(user_id: str, thread_id: str, request: ThreadUpdateRequest) -> ThreadHistoryResponse:
        try:
            return history_store.update_thread(user_id, thread_id, title=request.title, archived=request.archived)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/threads/{user_id}/{thread_id}")
    def delete_thread(user_id: str, thread_id: str) -> dict:
        try:
            history_store.update_thread(user_id, thread_id, archived=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"deleted": True, "user_id": user_id, "thread_id": thread_id}

    @app.get("/api/history/{user_id}", response_model=HistoryResponse)
    def user_history(user_id: str) -> HistoryResponse:
        return history_store.history(user_id)

    @app.delete("/api/history/{user_id}")
    def clear_user_history(user_id: str, thread_id: str = "default") -> dict:
        history_store.clear_history(user_id, thread_id)
        return {"cleared": True, "user_id": user_id, "thread_id": thread_id}

    @app.post("/api/cancel/{request_id}")
    def cancel_request(request_id: str) -> dict:
        event = cancel_event_for(request_id)
        event.set()
        with progress_lock:
            record = agent_progress.get(request_id)
        if record:
            steps = [AgentStep(**step) for step in record["steps"]]
            for step in steps:
                if step.status == "active":
                    step.status = "error"
                    step.detail = "キャンセルされました"
            if steps:
                steps[-1].status = "error"
                steps[-1].detail = "ユーザーがリクエストをキャンセルしました"
            set_steps(request_id, steps, done=True)
        return {"cancelled": True, "request_id": request_id}

    def execute_chat_job(request_id: str, request: ChatRequest) -> None:
        cancel_event = cancel_event_for(request_id)
        steps = initial_steps(request.agent_mode)
        set_steps(request_id, steps)
        set_async_job(request_id, done=False, error=None)
        resolved_mode = "chat"
        search_decision = ""
        sources: list[SearchResult] = []
        try:
            if engine.is_loaded and engine.active_model_id != request.model_id:
                raise RuntimeError("Select the model in the UI first. The server restarts to switch models cleanly.")
            cursor = 1
            resolved_mode, search_decision = resolve_agent_mode(request)
            if request.agent_mode == "auto":
                update_step(request_id, steps, cursor, "active", "検索が必要か判定しています")
                update_step(request_id, steps, cursor, "done", search_decision)
                cursor += 1
            if resolved_mode in {"web", "deep"}:
                search_index = insert_search_steps(steps, resolved_mode)
                if search_index:
                    cursor = search_index
                    set_steps(request_id, steps)
                update_step(request_id, steps, cursor, "active", "検索クエリを実行しています")
                search_bundle = collect_sources(
                    request.messages[-1].content,
                    resolved_mode,
                    cancel_event=cancel_event,
                )
                sources = search_bundle.sources
                detail = f"{len(sources)} 件の候補を取得"
                if search_bundle.warnings:
                    detail = f"{len(sources)} 件取得。一部検索失敗: {len(search_bundle.warnings)} 件"
                update_step(request_id, steps, cursor, "done", detail)
                cursor += 1
                update_step(request_id, steps, cursor, "active", "モデルへ渡す根拠を整えています")
                if sources:
                    update_step(request_id, steps, cursor, "done", f"{min(len(sources), 8)} 件をコンテキスト化")
                else:
                    update_step(request_id, steps, cursor, "done", "検索結果なし。通常プロンプトで続行")
                cursor += 1

            update_step(request_id, steps, cursor, "active", "会話履歴とエージェント結果を結合しています")
            enriched_request = request_with_sources(request, sources)
            if resolved_mode == "deep" and enriched_request.max_new_tokens is None:
                enriched_request = enriched_request.model_copy(update={"max_new_tokens": 4096})
            update_step(request_id, steps, cursor, "done", "生成用プロンプトを作成")
            cursor += 1
            update_step(request_id, steps, cursor, "active", "HPU 上のモデルで生成しています")
            response = engine.generate(enriched_request, sources=sources, cancel_event=cancel_event)
            response = response.model_copy(
                update={
                    "agent_mode": request.agent_mode,
                    "resolved_mode": resolved_mode,
                    "search_decision": search_decision,
                }
            )
            history_store.append_message(
                request.user_id,
                request.thread_id,
                ChatMessage(role="assistant", content=response.reply),
                last_mode=request.agent_mode,
                last_model_id=request.model_id,
            )
            steps[-2].status = "done"
            steps[-2].detail = "生成が完了しました"
            steps[-1].status = "done"
            steps[-1].detail = "ブラウザへ応答を返しました"
            set_steps(request_id, steps, done=True)
            set_async_job(request_id, done=True, response=response.model_dump(), error=None)
            print(
                f"Async chat job complete: request_id={request_id}, user={request.user_id}, "
                f"thread={request.thread_id}, resolved={resolved_mode}",
                flush=True,
            )
        except RequestCancelled:
            for step in steps:
                if step.status == "active":
                    step.status = "error"
                    step.detail = "キャンセルされました"
            steps[-1].status = "error"
            steps[-1].detail = "ユーザーがリクエストをキャンセルしました"
            set_steps(request_id, steps, done=True)
            set_async_job(request_id, done=True, error="Request cancelled")
        except Exception as exc:
            update_step(request_id, steps, len(steps) - 1, "error", str(exc), done=True)
            set_async_job(request_id, done=True, error=str(exc))
            print(f"Async chat job error: request_id={request_id}, error={exc}", flush=True)

    @app.post("/api/chat/jobs")
    def enqueue_chat_job(request: ChatRequest) -> dict:
        request_id = request_key(request.request_id)
        cancel_event = cancel_event_for(request_id)
        cancel_event.clear()
        set_progress_thread(request_id, request.user_id, request.thread_id)
        with progress_lock:
            if request_id in async_jobs:
                return {"request_id": request_id, "queued": True}
            async_jobs[request_id] = {
                "request_id": request_id,
                "done": False,
                "response": None,
                "error": None,
                "updated_at": time.time(),
            }
        history_store.append_message(
            request.user_id,
            request.thread_id,
            request.messages[-1],
            last_mode=request.agent_mode,
            last_model_id=request.model_id,
        )
        worker = threading.Thread(target=execute_chat_job, args=(request_id, request), daemon=True)
        worker.start()
        return {"request_id": request_id, "queued": True}

    @app.get("/api/chat/jobs/{request_id}", response_model=AsyncJobResponse)
    def chat_job_status(request_id: str) -> AsyncJobResponse:
        with progress_lock:
            job = async_jobs.get(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        response = ChatResponse(**job["response"]) if job.get("response") else None
        return AsyncJobResponse(
            request_id=request_id,
            done=bool(job.get("done")),
            response=response,
            error=job.get("error"),
            updated_at=float(job.get("updated_at", time.time())),
        )

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        request_id = request_key(request.request_id)
        cancel_event = cancel_event_for(request_id)
        cancel_event.clear()
        set_progress_thread(request_id, request.user_id, request.thread_id)
        steps = initial_steps(request.agent_mode)
        set_steps(request_id, steps)
        resolved_mode = "chat"
        search_decision = ""
        try:
            if engine.is_loaded and engine.active_model_id != request.model_id:
                raise HTTPException(
                    status_code=409,
                    detail="Select the model in the UI first. The server restarts to switch models cleanly.",
                )
            cursor = 1
            resolved_mode, search_decision = resolve_agent_mode(request)
            if request.agent_mode == "auto":
                update_step(request_id, steps, cursor, "active", "検索が必要か判定しています")
                update_step(request_id, steps, cursor, "done", search_decision)
                cursor += 1
            if resolved_mode in {"web", "deep"}:
                search_index = insert_search_steps(steps, resolved_mode)
                if search_index:
                    cursor = search_index
                    set_steps(request_id, steps)
                update_step(request_id, steps, cursor, "active", "検索クエリを実行しています")
                search_bundle = collect_sources(
                    request.messages[-1].content,
                    resolved_mode,
                    cancel_event=cancel_event,
                )
                sources = search_bundle.sources
                if search_bundle.warnings:
                    detail = f"{len(sources)} 件取得。一部検索失敗: {len(search_bundle.warnings)} 件"
                else:
                    detail = f"{len(sources)} 件の候補を取得"
                update_step(request_id, steps, cursor, "done", detail)
                cursor += 1
                update_step(request_id, steps, cursor, "active", "モデルへ渡す根拠を整えています")
                if cancel_event.is_set():
                    raise RequestCancelled()
                if sources:
                    update_step(request_id, steps, cursor, "done", f"{min(len(sources), 8)} 件をコンテキスト化")
                else:
                    update_step(request_id, steps, cursor, "done", "検索結果なし。通常プロンプトで続行")
                cursor += 1
            else:
                sources = []
            update_step(request_id, steps, cursor, "active", "会話履歴とエージェント結果を結合しています")
            if cancel_event.is_set():
                raise RequestCancelled()
            enriched_request = request_with_sources(request, sources)
            if resolved_mode == "deep" and enriched_request.max_new_tokens is None:
                enriched_request = enriched_request.model_copy(update={"max_new_tokens": 4096})
            update_step(request_id, steps, cursor, "done", "生成用プロンプトを作成")
            cursor += 1
            update_step(request_id, steps, cursor, "active", "HPU 上のモデルで生成しています")
            response = engine.generate(enriched_request, sources=sources, cancel_event=cancel_event)
            saved_messages = list(request.messages)
            saved_messages.append(ChatMessage(role="assistant", content=response.reply))
            history_store.replace_history(
                request.user_id,
                saved_messages,
                request.thread_id,
                last_mode=request.agent_mode,
                last_model_id=request.model_id,
            )
            return response.model_copy(
                update={"agent_mode": request.agent_mode, "resolved_mode": resolved_mode, "search_decision": search_decision}
            )
        except HTTPException:
            update_step(request_id, steps, len(steps) - 1, "error", "リクエストを完了できませんでした", done=True)
            raise
        except RequestCancelled as exc:
            for step in steps:
                if step.status == "active":
                    step.status = "error"
                    step.detail = "キャンセルされました"
            steps[-1].status = "error"
            steps[-1].detail = "ユーザーがリクエストをキャンセルしました"
            set_steps(request_id, steps, done=True)
            raise HTTPException(status_code=499, detail="Request cancelled") from exc
        except requests.RequestException as exc:
            update_step(request_id, steps, len(steps) - 1, "error", f"Web検索に失敗しました: {exc}", done=True)
            raise HTTPException(status_code=502, detail=f"Web search failed: {exc}") from exc
        except Exception as exc:
            update_step(request_id, steps, len(steps) - 1, "error", str(exc), done=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            with progress_lock:
                record = agent_progress.get(request_id)
            if record and not record["done"]:
                steps[-2].status = "done"
                steps[-2].detail = "生成が完了しました"
                steps[-1].status = "done"
                steps[-1].detail = "ブラウザへ応答を返しました"
                set_steps(request_id, steps, done=True)

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        request_id = request_key(request.request_id)
        cancel_event = cancel_event_for(request_id)
        cancel_event.clear()
        set_progress_thread(request_id, request.user_id, request.thread_id)
        steps = initial_steps(request.agent_mode)
        set_steps(request_id, steps)
        try:
            if engine.is_loaded and engine.active_model_id != request.model_id:
                raise HTTPException(
                    status_code=409,
                    detail="Select the model in the UI first. The server restarts to switch models cleanly.",
                )
        except HTTPException:
            update_step(request_id, steps, len(steps) - 1, "error", "リクエストを完了できませんでした", done=True)
            raise

        def line(event: dict) -> str:
            return json.dumps(event, ensure_ascii=False) + "\n"

        def status_event(message: str, status: str = "working") -> str:
            return line(
                {
                    "type": "status",
                    "message": message,
                    "status": status,
                    "steps": [step.model_dump() for step in steps],
                }
            )

        def stream_events():
            final_sent = False
            sources: list[SearchResult] = []
            resolved_mode = "chat"
            search_decision = ""
            try:
                print(
                    f"Chat request start: request_id={request_id}, user={request.user_id}, "
                    f"mode={request.agent_mode}, model={request.model_id}, messages={len(request.messages)}",
                    flush=True,
                )
                yield status_event(f"{AGENT_MODES[request.agent_mode]['label']} を開始しています")

                cursor = 1
                resolved_mode, search_decision = resolve_agent_mode(request)
                if request.agent_mode == "auto":
                    update_step(request_id, steps, cursor, "active", "検索が必要か判定しています")
                    yield status_event("検索要否を判定しています")
                    update_step(request_id, steps, cursor, "done", search_decision)
                    yield status_event(search_decision)
                    cursor += 1
                if resolved_mode in {"web", "deep"}:
                    search_index = insert_search_steps(steps, resolved_mode)
                    if search_index:
                        cursor = search_index
                        set_steps(request_id, steps)
                    queries = search_queries(request.messages[-1].content, resolved_mode)
                    update_step(
                        request_id,
                        steps,
                        cursor,
                        "active",
                        f"{len(queries)} 個の検索クエリを実行しています",
                    )
                    print(
                        f"Search start: request_id={request_id}, mode={resolved_mode}, queries={queries}",
                        flush=True,
                    )
                    yield status_event("自動検索中です" if resolved_mode == "web" else "Deep search 中です")
                    search_bundle = collect_sources(
                        request.messages[-1].content,
                        resolved_mode,
                        cancel_event=cancel_event,
                    )
                    sources = search_bundle.sources
                    if search_bundle.warnings:
                        detail = f"{len(sources)} 件取得。一部検索失敗: {len(search_bundle.warnings)} 件"
                    else:
                        detail = f"{len(sources)} 件の候補を取得"
                    update_step(request_id, steps, cursor, "done", detail)
                    print(
                        f"Search complete: request_id={request_id}, sources={len(sources)}, "
                        f"warnings={len(search_bundle.warnings)}",
                        flush=True,
                    )
                    yield status_event(f"検索完了: {detail}")
                    cursor += 1

                    update_step(request_id, steps, cursor, "active", "モデルへ渡す根拠を整えています")
                    if cancel_event.is_set():
                        raise RequestCancelled()
                    if sources:
                        update_step(request_id, steps, cursor, "done", f"{min(len(sources), 8)} 件をコンテキスト化")
                    else:
                        update_step(request_id, steps, cursor, "done", "検索結果なし。通常プロンプトで続行")
                    yield status_event("検索結果をプロンプトへ反映しました")
                    cursor += 1

                update_step(request_id, steps, cursor, "active", "会話履歴とエージェント結果を結合しています")
                if cancel_event.is_set():
                    raise RequestCancelled()
                enriched_request = request_with_sources(request, sources)
                if resolved_mode == "deep" and enriched_request.max_new_tokens is None:
                    enriched_request = enriched_request.model_copy(update={"max_new_tokens": 4096})
                update_step(request_id, steps, cursor, "done", "生成用プロンプトを作成")
                cursor += 1
                update_step(request_id, steps, cursor, "active", "HPU 上のモデルで生成しています")
                print(f"Generation start: request_id={request_id}, mode={request.agent_mode}", flush=True)
                yield status_event("モデル生成中です")
                if not engine.is_loaded or engine.active_model_id != enriched_request.model_id:
                    update_step(
                        request_id,
                        steps,
                        cursor,
                        "active",
                        f"初回モデルロード中です: {enriched_request.model_id}",
                    )
                    print(
                        f"Model load pending before generation: request_id={request_id}, "
                        f"model={enriched_request.model_id}",
                        flush=True,
                    )
                    yield status_event(f"初回モデルロード中です: {enriched_request.model_id}")

                for event in engine.stream_generate(enriched_request, sources=sources, cancel_event=cancel_event):
                    if event["type"] == "final":
                        response = ChatResponse(**event["data"])
                        saved_messages = list(request.messages)
                        saved_messages.append(ChatMessage(role="assistant", content=response.reply))
                        history_store.replace_history(
                            request.user_id,
                            saved_messages,
                            request.thread_id,
                            last_mode=request.agent_mode,
                            last_model_id=request.model_id,
                        )
                        steps[-2].status = "done"
                        steps[-2].detail = "生成が完了しました"
                        steps[-1].status = "done"
                        steps[-1].detail = "ブラウザへ応答を返しました"
                        set_steps(request_id, steps, done=True)
                        final_sent = True
                        print(
                            f"Chat request complete: request_id={request_id}, mode={request.agent_mode}, "
                            f"resolved={resolved_mode}, "
                            f"tokens={response.generated_tokens}, elapsed={response.elapsed_sec:.2f}s",
                            flush=True,
                        )
                        event["data"] = response.model_copy(
                            update={
                                "agent_mode": request.agent_mode,
                                "resolved_mode": resolved_mode,
                                "search_decision": search_decision,
                            }
                        ).model_dump()
                    yield line(event)
            except RequestCancelled:
                for step in steps:
                    if step.status == "active":
                        step.status = "error"
                        step.detail = "キャンセルされました"
                steps[-1].status = "error"
                steps[-1].detail = "ユーザーがリクエストをキャンセルしました"
                set_steps(request_id, steps, done=True)
                print(f"Chat request cancelled: request_id={request_id}", flush=True)
                yield line({"type": "error", "detail": "Request cancelled"})
            except requests.RequestException as exc:
                update_step(request_id, steps, len(steps) - 1, "error", f"Web検索に失敗しました: {exc}", done=True)
                print(f"Chat request search error: request_id={request_id}, error={exc}", flush=True)
                yield line({"type": "error", "detail": f"Web search failed: {exc}"})
            except Exception as exc:
                update_step(request_id, steps, len(steps) - 1, "error", str(exc), done=True)
                print(f"Chat request error: request_id={request_id}, error={exc}", flush=True)
                yield line({"type": "error", "detail": str(exc)})
            finally:
                with progress_lock:
                    record = agent_progress.get(request_id)
                if record and not record["done"] and not final_sent:
                    update_step(request_id, steps, len(steps) - 1, "error", "ストリームが中断されました", done=True)

        return StreamingResponse(stream_events(), media_type="application/x-ndjson")

    @app.post("/api/switch_model")
    def switch_model(request: SwitchModelRequest) -> dict:
        if request.model_id not in supported_model_specs():
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model_id}")
        if not engine.is_loaded:
            engine.default_model_id = request.model_id
            return {
                "restarting": False,
                "active_model_id": None,
                "default_model_id": engine.default_model_id,
            }
        if engine.is_loaded and engine.active_model_id == request.model_id:
            engine.default_model_id = request.model_id
            return {
                "restarting": False,
                "active_model_id": engine.active_model_id,
                "default_model_id": engine.default_model_id,
            }

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
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID), choices=sorted(supported_model_specs()))
    parser.add_argument("--host", default=os.environ.get("SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    args.port = args.port or int(os.environ.get("SERVER_PORT", DEFAULT_SERVER_PORT))
    return args


app = create_app(os.environ.get("MODEL_ID", DEFAULT_MODEL_ID))


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    app = create_app(args.model_id)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
