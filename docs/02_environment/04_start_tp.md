# TP起動

235Bはtorchrunで起動する。

```bash
PATH=/home/test1/habanalabs-venv-optimum/bin:$PATH \
HF_HOME=$PWD/hf_cache \
/home/test1/habanalabs-venv-optimum/bin/python -m torch.distributed.run \
 --standalone --nproc_per_node=8 chat_server.py \
 --host 0.0.0.0 --tensor-parallel-size 8
```

