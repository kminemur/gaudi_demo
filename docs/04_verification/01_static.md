# 静的検証

変更後は必ず実行する。

```bash
python3 -m py_compile chat_server.py
```

docs変更時は各Markdownが400文字以下か確認する。

```bash
find docs -type f -name '*.md' -exec wc -m {} +
```

