# HPU設定

既定値は `PT_HPU_LAZY_MODE=0`、`PT_HPU_WEIGHT_SHARING=0`。モデルロード時に `htcore.hpu_inference_set_env()` を呼ぶ。Optimum Habana が使える場合は `adapt_transformers_to_gaudi()` を適用する。

