# スレッド要件

会話はユーザーごとの複数スレッドで分離する。`ChatRequest.messages` は選択中スレッドのみ含める。別トピックの履歴を混ぜない。スレッドは `thread_id,title,messages,last_mode,last_model_id,running` を持ち、処理中状態をUIへ返す。

