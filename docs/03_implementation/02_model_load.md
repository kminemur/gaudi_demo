# モデルロード

TP時は `device_map` を使わない。`from_pretrained()` へ `tp_plan=get_tensor_parallel_plan(model_id)` と `tp_size` を渡す。入力テンソルは `"hpu"` へ送る。`hpu:0` のような添字付きHPU名はTP経路で使わない。

