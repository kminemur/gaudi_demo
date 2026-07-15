# 運用制約

235B BF16はメモリ負荷が高い。Gaudiでは `device_map` multi-HPUではなくtensor parallelを使う。TP中のストリーミングは一括生成フォールバックでよい。連続ジョブは受け付けるが、生成は直列実行する。

