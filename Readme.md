# Gaudi demo

Run Qwen/Qwen3.6-27B on Intel Gaudi HPU.

```bash
HF_HOME=$PWD/hf_cache /home/test1/habanalabs-venv/bin/python run_qwen36_hpu.py \
  --prompt "日本語で短く自己紹介して"
```

This machine has Habana software 1.24 installed. The working Python environment is
`/home/test1/habanalabs-venv`.

Note: `Qwen/Qwen3.6-27B` requires a newer Transformers release with `qwen3_5`
support. `optimum-habana==1.21.0` is installed in the environment, but its public
wrappers are currently pinned to `transformers<4.56`, so this demo uses the
Habana PyTorch bridge directly while keeping the Gaudi/Optimum Habana environment
installed.
