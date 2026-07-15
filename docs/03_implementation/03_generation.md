# 生成処理

rank0は検索とプロンプト作成後、`ChatRequest` を全rankへbroadcastする。全rankは同じトークナイズ、入力転送、`model.generate()` を実行する。rank0だけが応答文、metrics、履歴保存、UI応答を担当する。

