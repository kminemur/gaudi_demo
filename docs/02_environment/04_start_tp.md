# TP起動

235Bサーバーは簡易スクリプトで起動する。

```bash
./start_chat_server.sh
```

既定は8 HPU、port 8000、`hf_cache`。変更例:

```bash
SERVER_PORT=8080 CHAT_TENSOR_PARALLEL_SIZE=8 ./start_chat_server.sh
```
