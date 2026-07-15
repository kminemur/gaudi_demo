# 起動検証

torchrun起動ログで確認する:
- rank0だけがUvicornを起動
- 全rankがモデルロードに参加
- `tensor_parallel_size=8`
- `tp_plan` 経路を使用
- `"hpu:X" notation is not supported` 警告が出ない

