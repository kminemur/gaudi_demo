# 非同期ジョブ

UI標準経路は `POST /api/chat/jobs`。即時に `request_id` を返し、UIは `GET /api/chat/jobs/{id}` と `GET /api/agent_steps/{id}` をpollする。HPU生成は `ChatEngine.lock` で直列化する。

