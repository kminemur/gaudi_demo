# vLLM Gaudi LLM server demo

Intel Gaudi 上に、OpenAI 互換の LLM API サーバーを構築する最小デモです。
[vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) の公式手順に沿って、Gaudi環境に対応するvLLMとプラグインをソースから導入します。

## 必要な環境

- Intel Gaudi Software 1.24.1が導入済みのLinux
- Python 3、Git、curl
- モデルにアクセスするための `HF_TOKEN`（必要な場合のみ）

## 1. インストール

```bash
bash setup.sh
```

既存の仮想環境を使う場合は `PYTHON` を指定します。

```bash
PYTHON=/path/to/venv/bin/python bash setup.sh
```

`setup.sh` はPyTorchから互換バージョンを自動選択し、Habana用PyTorchを保持します。

| PyTorch | Gaudi Software | vLLM Gaudi |
| --- | --- | --- |
| 2.11 | 1.24.1 | 0.24.0 |

vLLM本体には、vLLM Gaudi 0.24.0が指定する検証済みコミットを使用します。

CUDA版PyTorchが入っている環境では処理を停止します。事前確認:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c \
  'import torch; print(torch.__version__, torch.version.cuda)'
python -c \
  'import torch, habana_frameworks.torch; print(torch.hpu.is_available())'
```

期待値はPyTorch `2.11.x`、CUDA `None`、HPU `True` です。CUDA版PyTorchやPyTorch 2.10が表示された場合は、その仮想環境をIntel Gaudi Software 1.24.1のインストーラーまたは公式コンテナから作り直してください。PyPIの `torch` だけを再インストールしてもHabanaランタイムとの整合性は戻りません。

## 2. サーバー起動

```bash
bash serve.sh
```

既定値は `Qwen/Qwen3-Coder-Next-FP8`、4 HPU、最大コンテキスト長
`262144`、ポート `8000` です。

```bash
MODEL=Qwen/Qwen3-Coder-Next-FP8 TP_SIZE=4 bash serve.sh
```

このFP8モデルの量子化ブロック幅は128、共有エキスパート幅は512のため、
`TP_SIZE` には `1`、`2`、`4` のいずれかを指定してください。`8` は各ランクの
入力幅が64となり、モデルのロードに失敗します。

主な設定は `MODEL`、`TP_SIZE`、`MAX_MODEL_LEN`、`HOST`、`PORT`、`API_KEY` です。

## 3. 疎通確認

別のターミナルで実行します。

```bash
bash request.sh
```

`API_KEY` を設定してサーバーを起動した場合は、同じ値を `request.sh` にも渡してください。

```bash
API_KEY=demo-secret bash request.sh
```

API は `POST /v1/chat/completions` なので、OpenAI SDK からも `base_url=http://<host>:8000/v1` を指定して利用できます。
