# モデル要件

チャットサーバーは `Qwen/Qwen3-235B-A22B` を既定モデルとする。235BはGaudi上で tensor parallel を使う。`device_map` の `hpu:0` 形式はGaudi bridge非対応なので、multi-HPU用途で使わない。

