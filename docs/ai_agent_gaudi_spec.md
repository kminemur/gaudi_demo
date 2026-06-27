# AI Agent Specification with Intel Gaudi

このドキュメントは、Gaudi Qwen Chat を `Auto` / `Chat` / `Deep search`
を備えた AI エージェントとして運用・拡張するための仕様をまとめる。

## 目的

チャットボットを単純な LLM 応答だけでなく、ユーザーの質問に応じて外部検索を実行し、
検索結果を根拠として回答できる AI エージェントにする。モデル推論は Intel Gaudi HPU
上で実行し、Web UI では回答作成中のステップを表示して、何が進行中なのかをユーザーが
把握できるようにする。

## システム概要

- Web サーバー: FastAPI
- UI: サーバー組み込み HTML / JavaScript
- 推論エンジン: Transformers + Habana PyTorch bridge
- Gaudi 最適化: Optimum Habana の Gaudi patch が利用可能な場合に適用
- 推論デバイス: Intel Gaudi HPU
- 検索エンジン: DuckDuckGo lite via `r.jina.ai` を標準利用
- 代替検索: `SEARCH_ENGINE=bing`
- 履歴保存: `chat_history.json`
- 会話単位: ユーザーごとに複数スレッドを持つ
- 標準ポート: `8000`

## エージェントモード

### Auto

ユーザーが検索機能を明示的に選ばなくても、質問内容から検索の必要性を
判定して自動で検索を実行する。通常の会話や文章生成では検索せず、最新性・事実確認・外部情報が
必要な場合だけ Web 検索へ切り替える。

主な用途:

- ユーザーに検索モードを意識させない自然なチャット体験
- ニュース、天気、障害、株価、スポーツ結果など最新性が重要な質問
- 企業、製品、法律、ライブラリ仕様など変化し得る情報の確認
- 「調べて」「最新」「今日」「現在」「直近」などの依頼

処理ステップ:

1. リクエスト受付
2. 検索要否判定
3. 必要な場合は自動検索または Deep search
4. 検索結果の整理
5. プロンプト作成
6. モデル生成
7. 応答完了

Auto は UI の標準モードとする。ユーザーが明示的に `Chat` を選んだ場合は検索しない。
ユーザーが `Deep search` を選んだ場合は、自動判定を経由せず詳細検索を実行する。

### Chat

LLM のみで回答する。外部検索は行わない。

主な用途:

- 一般的な文章生成
- 翻訳
- 要約
- コードや設定の相談
- 検索が不要な質問

処理ステップ:

1. リクエスト受付
2. プロンプト作成
3. モデル生成
4. 応答完了

## 内部 Web search

Web search はユーザーに選択肢として表示しない。Auto モードの検索要否判定で必要と判断された
場合に、内部処理として 1 つの検索クエリを実行し、取得した検索結果をプロンプトに追加して回答する。

主な用途:

- 最新情報の確認
- 事実確認
- ニュース、障害状況、製品情報など、変化する情報の参照

検索仕様:

- 検索クエリ数: 1
- 1 クエリあたりの取得候補: 最大 5 件
- 回答に渡すソース数: 最大 5 件

内部処理ステップ:

1. リクエスト受付
2. 自動検索
3. 検索結果の整理
4. プロンプト作成
5. モデル生成
6. 応答完了

### Deep search

ユーザー入力から複数の検索クエリを作成し、広めに情報を集めて回答する。

主な用途:

- 詳細調査
- 比較
- 背景説明が必要な質問
- 複数ソースで裏取りしたい質問

検索仕様:

- 検索クエリ:
  - 元の質問
  - 元の質問 + `最新`
  - 元の質問 + `背景 解説`
- 1 クエリあたりの取得候補: 最大 4 件
- 回答に渡すソース数: 最大 8 件
- 最小生成トークン上限: 4096

Deep search は長めの回答を想定するため、通常の reasoning preset よりも
`max_new_tokens` が小さい場合は 4096 を優先する。

## 自動 Web search 判定

Auto モードでは、ユーザーの最新メッセージと現在のスレッド文脈を使って検索の必要性を判定する。
初期実装ではルールベースを基本とし、将来的には軽量な LLM 判定へ拡張できるようにする。

### 検索する条件

以下のいずれかに該当する場合は自動検索を実行する。

- 最新性が必要な質問
- ニュース、速報、障害、イベント、スポーツ結果、天気、株価、為替、価格、リリース情報
- 法律、規制、仕様、API、ライブラリ、製品情報など更新される可能性がある話題
- 「今日」「現在」「今」「最新」「直近」「今年」「昨日」「明日」など相対日付を含む質問
- 「検索して」「調べて」「ソース付きで」「根拠を出して」など明示的な検索依頼
- 回答に外部 URL、公式情報、出典提示が必要な質問

### 検索しない条件

以下の場合は LLM のみで回答する。

- 雑談、文章作成、翻訳、要約、アイデア出し
- ユーザーが提供したテキストだけで完結する質問
- コード説明や一般的な設計相談
- ユーザーが明示的に「検索しないで」と指定した場合

### 自動検索と Deep search の自動選択

Auto モードで検索が必要と判定した場合、基本は内部 Web search による自動検索を使う。
以下の場合は Deep search を選ぶ。

- 比較、調査、複数候補の検討
- 背景説明が必要な話題
- 「詳しく」「深く調べて」「網羅的に」「複数ソースで」などの依頼
- 速報ではなく、複数観点の整理が必要な質問

判定結果は UI のステップに表示する。

```text
検索要否判定: 最新情報が必要なため自動検索を実行
検索要否判定: 複数ソースでの比較が必要なため Deep search を実行
検索要否判定: 検索不要。モデルのみで回答
```

## 検索結果の扱い

検索結果は `SearchResult` として扱う。

```text
title   : 検索結果タイトル
url     : 検索結果 URL
snippet : 検索結果スニペット
```

検索結果は重複 URL を除外し、最大件数まで収集する。検索エラーが発生した場合でも、
他のクエリが成功していれば処理を継続する。

検索結果がある場合、最後のユーザーメッセージに以下の情報を追加してモデルへ渡す。

```text
<元の質問>

<モード名> の検索結果:
[1] <title>
URL: <url>
概要: <snippet>

上の検索結果を根拠として使い、必要なら [1] のように番号で出典を示して日本語で答えてください。
途中で切らず、長くなりそうな場合は要点を絞って最後まで完結させてください。
```

## スレッド仕様

チャットにはスレッドの概念を導入する。スレッドは 1 つの話題または作業単位を表し、
同じユーザーが複数のトピックを並行して会話できるようにする。

### 目的

- 別トピックの会話履歴が混ざらないようにする
- 複数の調査や作業を同時に進められるようにする
- 過去の会話を一覧から再開できるようにする
- Auto search の判定に、そのスレッドの文脈だけを使えるようにする

### Thread

スレッドは以下の情報を持つ。

```text
thread_id      : 一意な ID
user_id        : 所有ユーザー ID
title          : スレッド名
messages       : そのスレッド内の会話履歴
created_at     : 作成日時
updated_at     : 更新日時
last_mode      : 最後に使用した agent mode
last_model_id  : 最後に使用した model
archived       : 一覧から隠す場合 true
```

### スレッドタイトル

新規スレッド作成時は以下のいずれかでタイトルを決める。

- ユーザーが入力したタイトル
- 最初のユーザーメッセージの先頭 30-40 文字
- 将来的には LLM による短いタイトル生成

### 会話履歴の分離

`ChatRequest.messages` は選択中スレッドの履歴だけを含める。別スレッドの履歴はモデルへ渡さない。
これにより、トピック混線と不要なコンテキスト増大を避ける。

### 同時会話

ユーザーは複数スレッドを作成でき、画面上のスレッド一覧から切り替える。各スレッドは独立した
進捗状態を持つ。

例:

- スレッド A: 地震速報の確認
- スレッド B: World Cup 日本代表の結果調査
- スレッド C: Gaudi 推論性能の改善相談

スレッド A で自動検索中でも、スレッド B へ切り替えて履歴を確認できる。単一 HPU 構成では
モデル生成自体は `ChatEngine.lock` により直列化するが、UI 上は各スレッドの状態を個別に表示する。

## 推論モデル

利用可能なモデルは環境の Transformers 対応状況によって変わる。

| モデル | 種別 | 精度 | 備考 |
| --- | --- | --- | --- |
| `Qwen/Qwen3.6-27B` | image_text | bf16 | `qwen3_5` 対応 Transformers が必要 |
| `Qwen/Qwen3.6-27B-FP8` | image_text | fp8 pretrained | 現行 HPU 経路では bf16 へフォールバック |
| `Qwen/Qwen3.6-35B-A3B-FP8` | image_text | fp8 pretrained | `qwen3_5` 対応 Transformers が必要 |
| `Qwen/Qwen3-32B` | causal_lm | bf16 | Optimum Habana 互換 venv の標準 fallback |

デフォルトモデル:

- Transformers が `qwen3_5` を認識する場合: `Qwen/Qwen3.6-27B`
- 認識しない場合: `Qwen/Qwen3-32B`

## Intel Gaudi 推論仕様

### 実行方式

- 推論配置: single HPU Transformers path
- モデル配置: `device_map={"": "hpu"}`
- dtype:
  - bf16 モデル: `torch.bfloat16`
  - FP8 checkpoint: 実行可能な場合は checkpoint 指定、正しさに問題があるモデルは bf16 へ fallback
- KV cache: `use_cache=True`
- 入力テンソル: `USE_INT32_INPUTS=1` の場合、`input_ids` と `attention_mask` を int32 化
- 実行モード:
  - `PT_HPU_LAZY_MODE=1`: lazy
  - それ以外: eager

### 起動時の Habana 設定

アプリ起動時に以下を既定値として設定する。

```text
PT_HPU_LAZY_MODE=0
PT_HPU_WEIGHT_SHARING=0
```

モデルロード時に `htcore.hpu_inference_set_env()` を呼び出し、Optimum Habana が利用可能な
環境では `adapt_transformers_to_gaudi()` を適用する。

### モデルロード

サーバー起動時にはモデルをロードしない。最初のチャットリクエスト時に選択モデルをロードする。

ログには以下を出力する。

- モデルロード開始
- requested model
- execution model
- model kind
- precision
- モデルロード完了
- ロード時間

モデル切り替え時は現在のモデルを unload し、HPU cache を解放してから次のモデルをロードする。

## Reasoning Preset

| Preset | thinking | max_new_tokens | 用途 |
| --- | --- | ---: | --- |
| Low | disabled | 128 | 短い応答、速度優先 |
| Medium | disabled | 512 | 標準応答 |
| High | enabled | 1024 | 長めの推論 |

`Deep search` は最低 4096 token を確保する。

## ストリーミング

UI は `/api/chat/stream` を使い、JSON Lines 形式でイベントを受け取る。

イベント種別:

- `status`: 現在の処理状態、ステップ一覧
- `delta`: 生成済みテキスト差分
- `sources`: 検索ソース一覧
- `metrics`: elapsed, TTFT, tokens/sec, generated tokens
- `final`: 最終回答
- `error`: エラー

これにより、モデル生成の完了を待たずにユーザーへ進捗と部分回答を表示できる。

## 非同期ジョブ実行

連続推論を可能にするため、UI の標準送信経路は非同期ジョブ API を使う。ユーザーが
メッセージを送信すると、サーバーはリクエストをジョブとして受け付け、即座に `request_id` を返す。
UI は入力欄をすぐ再度有効化し、ユーザーは前の回答完了を待たずに次のメッセージを送信できる。

実行仕様:

- `POST /api/chat/jobs` でジョブを作成する
- `GET /api/chat/jobs/{request_id}` で完了状態と最終応答を取得する
- `GET /api/agent_steps/{request_id}` で処理ステップを取得する
- HPU 上のモデル生成は `ChatEngine.lock` により安全に直列化する
- 検索、プロンプト作成、モデル生成の進捗はジョブごとに保持する
- スレッド一覧には処理中スレッドを `running=true` として返す
- UI はページ全体を自動 refresh しない
- `updated_at` が変わったジョブ、ステップ、スレッド一覧だけを差分反映する

この方式により、同一スレッドまたは別スレッドで複数の推論を連続投入できる。single HPU 構成では
実際のモデル生成は順番に実行されるが、ユーザー操作はブロックしない。

## 非 JavaScript フォールバック

JavaScript が動作しないブラウザ環境でも `/chat/send` の通常フォーム送信で利用できる。

フォールバック動作:

1. ユーザー発話を履歴へ保存
2. バックグラウンドスレッドで検索・推論を開始
3. `/?job_id=<id>` にリダイレクト
4. ページ全体の自動 refresh は行わず、受付時点の進捗を表示
5. JavaScript が利用可能な場合は `updated_at` が変わった情報だけを差分反映
6. 完了後、回答を履歴へ保存

この経路でも画面に以下を表示する。

- 回答作成中
- 各ステップの状態
- 現在の詳細
- 経過秒数

## API

### `GET /`

チャット UI を返す。ログイン cookie がある場合はチャット画面を表示する。

### `GET /login?display_name=<name>`

JavaScript が動作しない場合のログイン fallback。cookie を設定して `/` へ戻す。

### `GET /logout`

cookie を削除してログアウトする。

### `GET /api/health`

モデル、検索、HPU 実行設定、ロード状態を返す。

主なレスポンス項目:

- `default_model_id`
- `models`
- `reasoning_presets`
- `agent_modes`
- `search_engine`
- `search_timeout_sec`
- `hpu_execution_mode`
- `use_int32_inputs`
- `model_placement`
- `active_model_id`
- `model_loaded`
- `precision`
- `loaded_at`

### `GET /api/threads/{user_id}`

ユーザーのスレッド一覧を返す。UI の左ペインまたはサイドバーで表示する。

主なレスポンス項目:

- `thread_id`
- `title`
- `message_count`
- `last_mode`
- `last_model_id`
- `updated_at`
- `running`

### `POST /api/threads`

新しいスレッドを作成する。

入力:

```text
user_id : ユーザー ID
title   : 任意。未指定の場合は最初のメッセージから自動生成
```

### `GET /api/threads/{user_id}/{thread_id}`

指定スレッドの会話履歴を返す。

### `PATCH /api/threads/{user_id}/{thread_id}`

スレッドタイトル変更、archive 状態変更などを行う。

### `DELETE /api/threads/{user_id}/{thread_id}`

指定スレッドを削除する。削除ではなく archive として隠す運用も選択できる。

### `POST /api/chat`

非ストリーミング推論 API。回答完了後に `ChatResponse` を返す。

### `POST /api/chat/stream`

ストリーミング推論 API。UI の標準経路。

### `POST /api/chat/jobs`

非同期推論ジョブを作成する。UI の標準経路。レスポンスには `request_id` と `queued` を返す。

### `GET /api/chat/jobs/{request_id}`

非同期推論ジョブの状態を返す。

主なレスポンス項目:

- `request_id`
- `done`
- `response`
- `error`
- `updated_at`

### `GET /api/agent_steps/{request_id}`

実行中リクエストのステップ状態を返す。

### `POST /api/cancel/{request_id}`

実行中リクエストへキャンセルを要求する。生成は次の token boundary で停止する。

### `GET /api/history/{user_id}`

後方互換用。スレッド導入後は active thread または default thread の履歴を返す。

### `DELETE /api/history/{user_id}`

後方互換用。スレッド導入後は active thread または default thread の履歴を削除する。

## データモデル

### ChatRequest

```text
request_id       : 任意。進捗管理用 ID
user_id          : ユーザー ID
thread_id        : スレッド ID
model_id         : 使用モデル
messages         : 会話履歴
reasoning_effort : low / medium / high
agent_mode       : auto / chat / deep
max_new_tokens   : 任意。1-4096
temperature      : 0.0-2.0
top_p            : 0.05-1.0
enable_thinking  : 任意。preset を上書き
```

### ThreadSummary

```text
thread_id     : スレッド ID
title         : スレッド名
message_count : メッセージ数
last_mode     : 最後に使った agent mode
last_model_id : 最後に使った model
updated_at    : 最終更新日時
running       : このスレッドで処理中の request があるか
```

### ThreadHistory

```text
user_id    : ユーザー ID
thread_id  : スレッド ID
title      : スレッド名
messages   : スレッド内の会話履歴
created_at : 作成日時
updated_at : 更新日時
```

### ChatResponse

```text
reply                    : 回答本文
precision                : 実行精度
reasoning_effort         : 使用 preset
agent_mode               : ユーザー指定 agent mode。auto / chat / deep
resolved_mode            : 実際に実行した内部 mode。chat / web / deep
search_decision          : Auto の検索要否判定結果
sources                  : 検索ソース
effective_max_new_tokens : 実際の生成上限
enable_thinking          : thinking 有効状態
elapsed_sec              : 推論時間
ttft_sec                 : first token までの時間
tokens_per_sec           : 生成速度
generated_tokens         : 生成 token 数
```

## UI 要件

チャット画面では、ユーザーが「何も起きていない」と感じないように以下を表示する。

- 接続状態
- モデルロード状態
- 送信可能状態
- 現在のスレッド名
- スレッド一覧
- 新規スレッド作成
- スレッド切り替え
- スレッド名変更
- スレッド削除または archive
- 回答作成中のステップ
- Auto search の検索要否判定
- 自動検索中であること
- 検索結果整理中であること
- HPU 上でモデル生成中であること
- 経過秒数
- 生成メトリクス
- ソースリンク
- キャンセルボタン

### スレッド UI

チャット画面にはスレッド一覧を左サイドバーとして表示する。モバイルではチャット上部に
縦並びで表示する。

スレッド一覧に表示する項目:

- スレッドタイトル
- 最終更新時刻
- 最後の agent mode
- 処理中インジケータ
- 未読または未表示の完了状態

スレッド操作:

- 新規スレッド作成
- スレッド選択
- 削除または archive
- 処理中スレッドへの復帰

スレッドタイトルは最初のユーザー発話から自動生成する。ユーザーに手動入力させない。

スレッドを切り替えた場合、入力欄、履歴、進捗表示、ソース表示は選択中スレッドのものに更新する。
実行中の別スレッドがある場合は、そのスレッドの一覧項目に `生成中` や `検索中` を表示する。

## 起動例

```bash
cd /home/test1/kazuki/gaudi_demo

HF_HOME=$PWD/hf_cache \
/home/test1/habanalabs-venv-optimum/bin/python chat_server.py \
  --host 0.0.0.0
```

アクセス先:

```text
http://<server-ip>:8000/
```

## 実装後の運用フロー

機能実装後は、動作確認まで行ったうえで git commit and push する。

標準フロー:

1. 変更対象を確認する。
2. 不要なログや一時ファイルを commit 対象から除外する。
3. 構文チェックまたは利用可能なテストを実行する。
4. 必要に応じてサーバーを再起動し、UI/API の動作を確認する。
5. `git diff` と `git status` で差分を確認する。
6. 実装内容が分かるメッセージで commit する。
7. 現在の branch を remote に push する。

実行例:

```bash
git status -sb
/home/test1/habanalabs-venv-optimum/bin/python -m py_compile chat_server.py
git add chat_server.py Readme.md docs/
git commit -m "Implement auto search threads"
git push -u origin "$(git branch --show-current)"
```

push に GitHub 認証が必要な場合は、認証エラーを隠さずに報告する。認証が未設定で push できない
場合でも、commit まで完了していれば commit hash と失敗理由を記録する。

## 環境変数

| 変数 | 既定値 | 説明 |
| --- | --- | --- |
| `SERVER_HOST` | `0.0.0.0` | bind host |
| `SERVER_PORT` | `8000` | listen port |
| `MODEL_ID` | 自動判定 | 起動時 default model |
| `CHAT_HISTORY_PATH` | `chat_history.json` | 履歴保存先 |
| `SEARCH_ENGINE` | `duckduckgo` | `duckduckgo` または `bing` |
| `SEARCH_TIMEOUT_SEC` | `8` | 検索 HTTP timeout |
| `AUTO_SEARCH_DEFAULT` | `1` | UI の標準モードを Auto search にする |
| `USE_INT32_INPUTS` | `1` | HPU 入力 tensor を int32 化 |
| `PT_HPU_LAZY_MODE` | `0` | HPU lazy/eager 切り替え |
| `PT_HPU_WEIGHT_SHARING` | `0` | 量子化モデルでの余分なメモリ消費を避ける |

## 現状の制約

- 現在の推論配置は single HPU Transformers path であり、8-card Gaudi2 全体を使う構成ではない。
- 高スループット運用では vLLM for Intel Gaudi、DeepSpeed、Optimum Habana tensor parallel の検討が必要。
- `qwen3_5` を必要とする Qwen3.6 系モデルは Transformers の対応バージョンに依存する。
- 一部 FP8 checkpoint は現行 HPU 経路で正しい logits を返さない場合があるため、bf16 sibling model へ fallback する。
- 検索は HTML 結果の簡易 parser に依存しているため、検索エンジン側の表示変更で parser 調整が必要になる可能性がある。
- 自動検索結果はスニペット中心であり、ページ本文全体の抽出やランキング再評価はまだ行っていない。
- Auto search の初期判定はルールベースとし、誤判定を UI から補正できるようにする。
- single HPU 構成では複数スレッドのモデル生成は直列化される。
- スレッド数が増えると履歴ファイルが大きくなるため、将来的には SQLite などへの移行を検討する。

## 拡張方針

AI エージェントとして強化する場合は、以下を優先する。

1. Auto search の判定精度改善
2. 検索結果本文の取得と要約
3. ソース品質スコアリング
4. 検索クエリ生成を LLM によって動的化
5. Deep search の反復検索
6. 回答前の根拠チェック
7. citation の厳密化
8. スレッド単位のツール実行ログ永続化
9. スレッド検索、並び替え、archive
10. 複数ユーザー・複数スレッド同時実行時のキュー制御
11. tensor parallel / multi HPU 推論
12. API 認証と監査ログ
