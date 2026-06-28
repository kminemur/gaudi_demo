# サーバー実装

rank0だけFastAPIを起動する。rank1以降はHTTPを持たず、モデルをロードしてworker loopで待機する。rank0はリクエストを受け、生成コマンドをdistributed broadcastし、全rankで同じ `generate()` に参加する。

