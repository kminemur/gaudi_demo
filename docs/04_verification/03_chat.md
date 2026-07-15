# チャット検証

UIで短い日本語質問を送る。期待値:
- ジョブがqueuedになる
- agent_stepsが進む
- 235Bが回答する
- 履歴へassistant応答が保存される
- `/api/health` が `model_loaded=true` を返す

