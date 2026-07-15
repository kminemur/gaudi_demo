import json
import unittest

from responses_api import (
    function_tools,
    make_response,
    output_items,
    parse_qwen_output,
    qwen_messages,
    stream_events,
    tool_name_map,
)


class ResponsesApiTests(unittest.TestCase):
    def test_translates_tool_round_trip(self):
        payload = {
            "instructions": "Use tools.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "list files"}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": "{\"cmd\":\"ls\"}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
            ],
        }
        messages = qwen_messages(payload)
        self.assertEqual(messages[0], {"role": "system", "content": "Use tools."})
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["name"], "shell")
        self.assertEqual(messages[-1]["content"], "a.txt")

    def test_normalizes_responses_function_schema(self):
        tools = function_tools(
            [{"type": "function", "name": "shell", "description": "Run", "parameters": {"type": "object"}}]
        )
        self.assertEqual(tools[0]["function"]["name"], "shell")

    def test_flattens_codex_namespace_tools(self):
        source = [{
            "type": "namespace",
            "name": "agents",
            "tools": [{"type": "function", "name": "spawn", "parameters": {"type": "object"}}],
        }]
        self.assertEqual(function_tools(source)[0]["function"]["name"], "agents__spawn")
        names = tool_name_map(source)
        text, calls = parse_qwen_output(
            '<tool_call>{"name":"agents__spawn","arguments":{}}</tool_call>', names
        )
        self.assertEqual(text, "")
        self.assertEqual(calls[0]["namespace"], "agents")
        self.assertEqual(output_items("", calls)[0]["name"], "spawn")

    def test_parses_qwen_tool_call(self):
        text, calls = parse_qwen_output(
            'Checking. <tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call>',
            {"shell"},
        )
        self.assertEqual(text, "Checking.")
        self.assertEqual(calls[0]["name"], "shell")
        self.assertEqual(json.loads(calls[0]["arguments"]), {"cmd": "pwd"})

    def test_emits_codex_stream_lifecycle(self):
        items = output_items("done", [{"name": "shell", "arguments": "{\"cmd\":\"pwd\"}"}])
        response = make_response({"model": "Qwen/Qwen3-32B"}, items, 10, 4)
        events = list(stream_events(response))
        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "response.created")
        self.assertIn("response.output_text.delta", event_types)
        self.assertIn("response.function_call_arguments.done", event_types)
        self.assertEqual(event_types[-1], "response.completed")
        added_message = next(
            event
            for event in events
            if event["type"] == "response.output_item.added" and event["item"]["type"] == "message"
        )
        self.assertEqual(added_message["item"]["content"][0]["type"], "output_text")


if __name__ == "__main__":
    unittest.main()
