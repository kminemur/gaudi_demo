"""Small OpenAI Responses API compatibility layer for the Qwen server."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Iterable


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def response_id(prefix: str = "resp") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") in {"input_text", "output_text", "text"}:
            parts.append(str(part.get("text", "")))
    return "\n".join(part for part in parts if part)


def tool_name_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    names: dict[str, dict[str, str]] = {}
    for tool in tools:
        if tool.get("type") == "function" and tool.get("name"):
            name = str(tool["name"])
            names[name] = {"name": name}
        elif tool.get("type") == "namespace" and tool.get("name"):
            namespace = str(tool["name"])
            for child in tool.get("tools") or []:
                if child.get("type") != "function" or not child.get("name"):
                    continue
                name = str(child["name"])
                names[f"{namespace}__{name}"] = {"name": name, "namespace": namespace}
    return names


def function_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the function schemas accepted by Qwen's chat template."""
    normalized = []
    candidates: list[tuple[dict[str, Any], str, str | None]] = []
    for tool in tools:
        if tool.get("type") == "function" and tool.get("name"):
            candidates.append((tool, str(tool["name"]), None))
        elif tool.get("type") == "namespace" and tool.get("name"):
            namespace = str(tool["name"])
            candidates.extend(
                (child, f"{namespace}__{child['name']}", namespace)
                for child in tool.get("tools") or []
                if child.get("type") == "function" and child.get("name")
            )
    for tool, qwen_name, namespace in candidates:
        description = str(tool.get("description", ""))
        if namespace:
            description = f"Codex namespace: {namespace}. {description}".strip()
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": qwen_name,
                    "description": description,
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return normalized


def qwen_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate Responses API input items into a Qwen chat-template conversation."""
    messages: list[dict[str, Any]] = []
    instructions = content_text(payload.get("instructions"))
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_value = payload.get("input", "")
    if isinstance(input_value, str):
        if input_value:
            messages.append({"role": "user", "content": input_value})
        return messages

    call_names: dict[str, str] = {}
    for item in input_value or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            name = str(item.get("name", ""))
            namespace = item.get("namespace")
            qwen_name = f"{namespace}__{name}" if namespace else name
            call_id = str(item.get("call_id", item.get("id", "")))
            call_names[call_id] = qwen_name
            arguments = item.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": call_id,
                            "function": {"name": qwen_name, "arguments": arguments},
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            call_id = str(item.get("call_id", ""))
            messages.append(
                {
                    "role": "tool",
                    "name": call_names.get(call_id, "tool"),
                    "tool_call_id": call_id,
                    "content": content_text(item.get("output")),
                }
            )
        elif item_type == "message" or item.get("role"):
            role = str(item.get("role", "user"))
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": content_text(item.get("content"))})
    return messages


def strip_thinking(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return "" if "<think>" in text else text


def parse_qwen_output(
    text: str,
    allowed_tools: set[str] | dict[str, dict[str, str]],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract Qwen ``<tool_call>`` blocks while preserving ordinary text."""
    text = strip_thinking(text)
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = str(value.get("name", ""))
        if name not in allowed_tools:
            continue
        arguments = value.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        target = allowed_tools[name] if isinstance(allowed_tools, dict) else {"name": name}
        calls.append({**target, "arguments": arguments})
    visible_text = TOOL_CALL_PATTERN.sub("", text).strip()
    return visible_text, calls


def output_items(text: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if text:
        items.append(
            {
                "id": response_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for call in calls:
        item = {
            "id": response_id("fc"),
            "type": "function_call",
            "status": "completed",
            "call_id": response_id("call"),
            "name": call["name"],
            "arguments": call["arguments"],
        }
        if call.get("namespace"):
            item["namespace"] = call["namespace"]
        items.append(item)
    return items


def make_response(
    request: dict[str, Any],
    items: list[dict[str, Any]],
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    return {
        "id": response_id(),
        "object": "response",
        "created_at": int(time.time()),
        "completed_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": request.get("instructions"),
        "max_output_tokens": request.get("max_output_tokens"),
        "model": request.get("model"),
        "output": items,
        "output_text": "".join(
            part.get("text", "")
            for item in items
            if item.get("type") == "message"
            for part in item.get("content", [])
        ),
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", True)),
        "previous_response_id": request.get("previous_response_id"),
        "reasoning": request.get("reasoning") or {"effort": None, "summary": None},
        "store": bool(request.get("store", False)),
        "temperature": request.get("temperature", 0.0),
        "text": request.get("text") or {"format": {"type": "text"}},
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools", []),
        "top_p": request.get("top_p", 1.0),
        "truncation": request.get("truncation", "disabled"),
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
        "metadata": request.get("metadata") or {},
    }


def stream_events(response: dict[str, Any]) -> Iterable[dict[str, Any]]:
    pending = {**response, "status": "in_progress", "completed_at": None, "output": [], "usage": None}
    yield {"type": "response.created", "response": pending}
    yield {"type": "response.in_progress", "response": pending}
    for output_index, item in enumerate(response["output"]):
        if item["type"] == "message":
            # Codex seeds its active assistant item from this content entry. An
            # empty content list follows the public example but leaves current
            # Codex clients without an active item for subsequent deltas.
            started = {
                **item,
                "status": "in_progress",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
            yield {"type": "response.output_item.added", "output_index": output_index, "item": started}
            part = item["content"][0]
            yield {
                "type": "response.content_part.added",
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": 0,
                "part": {**part, "text": ""},
            }
            yield {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": 0,
                "delta": part["text"],
            }
            yield {
                "type": "response.output_text.done",
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": 0,
                "text": part["text"],
            }
            yield {
                "type": "response.content_part.done",
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": 0,
                "part": part,
            }
        else:
            started = {**item, "status": "in_progress", "arguments": ""}
            yield {"type": "response.output_item.added", "output_index": output_index, "item": started}
            yield {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": output_index,
                "delta": item["arguments"],
            }
            yield {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": output_index,
                "arguments": item["arguments"],
            }
        yield {"type": "response.output_item.done", "output_index": output_index, "item": item}
    yield {"type": "response.completed", "response": response}


def sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
