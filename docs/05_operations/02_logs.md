# ログ運用

監視対象:
- モデルロード開始/完了
- tensor_parallel_size
- HPU/HCCLエラー
- OOM
- 検索失敗
- ジョブ完了/失敗

まず `tail -f chat_server.log` で起動から初回生成まで確認する。

