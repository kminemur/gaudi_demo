# 起動停止

起動はtorchrunを使う。停止は該当プロセスを終了する。

```bash
pkill -f "chat_server.py --host 0.0.0.0"
```

本番相当では `nohup` かサービス管理を使い、stdout/stderrを `chat_server.log` に保存する。

