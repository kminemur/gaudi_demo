# 目的

Gaudi Qwen Chat は、Intel Gaudi HPU 上で Qwen 235B を動かす Web チャットである。FastAPI UI、会話履歴、検索付き回答、非同期ジョブを備える。AIエージェントは、既存UIを壊さず、235B tensor parallel 推論を安定運用できるよう実装する。

