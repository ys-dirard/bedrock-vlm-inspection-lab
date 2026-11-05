# bedrock-vlm-inspection-lab

Amazon Bedrock の VLM（画像を読めるモデル）を外観検査に使うときの、精度・費用・信頼性を公開データで測る検証基盤です。

- **一度取った応答は全部保存し、集計は後から何度でもやり直します。** 応答の本文（raw_text）から読み直すので、読み取りの規則や指標を変えても API を呼び直さなくて済みます
- **データ、プロンプト、モデル、呼び方は設定ファイル（YAML）で切り替えます**
- **費用はトークン数 × 単価で記録します。** 単価には出典 URL と確認日を付けます（`pricing.yaml`）
- **乱数のシードを固定し（random_state=42）、使った画像 ID の一覧を公開します**（`data/splits/`）

## 構成

```
bedrock-vlm-inspection-lab/
├── data/
│   ├── README.md           VisA の取得方法、出典とライセンス
│   └── splits/             抽出した画像 ID の一覧（CSV）
├── prompts/                プロンプト（版ごとに残す）
├── configs/                実験ごとの設定（YAML）
├── docs/contracts.md       読み取り・指標・費用の契約（単体テストと対応）
├── src/vlmlab/
│   ├── data.py             VisA の取得・抽出・画像の読み込みと縮小
│   ├── bedrock.py          Converse の呼び出し、再試行、usage とレイテンシの記録
│   ├── run.py              設定ファイルを読んで実験を回し、JSONL に追記
│   ├── parse.py            応答の JSON を読み取る（救済はしない）
│   ├── metrics.py          評価指標
│   ├── cost.py             単価表と費用の計算、累計費用での停止
│   ├── report.py           集計表（CSV と Markdown）を出す
│   └── cli.py              コマンドライン（vlmlab）
├── tests/                  単体テスト
├── pricing.yaml            単価表
└── runs/                   実行結果（コミットしない）
```

## 準備

[uv](https://docs.astral.sh/uv/) を使います。

```bash
uv sync
uv run pytest            # 単体テスト（AWS は呼ばない）
uv run vlmlab fetch      # VisA を取得・展開（認証なし、約1.9 GB）
uv run vlmlab splits     # data/splits/ を作り直す（コミット済みのものと同じになる）
```

Bedrock を呼ぶには、AWS の認証情報（`AWS_PROFILE` など）と、使うモデルへの `bedrock:InvokeModel` の権限が要ります。

## 使い方

### 1. 設定を書く

```yaml
# configs/n1_B.yaml の例
name: n1_B
split: data/splits/visa_test_200.csv
refs: data/splits/visa_refs.csv     # 良品の見本（テストと重ならない）
n_refs: 1                           # 見本の枚数（0 なら見本なし）
model_id: jp.amazon.nova-2-lite-v1:0
region: ap-northeast-1
service_tier: standard              # standard / flex / priority
prompt: prompts/v1_judge.txt
image_long_edge: null               # null なら原寸。768 などにすると長辺をそろえて縮める
inference: {temperature: 0, maxTokens: 300}
repeats: 3
budget_usd: 2.0                     # この実験の累計がこれを超えそうなら止める
parse:                              # 省略すると {"judgement": "OK"/"NG", "confidence": 0〜1}
  judgement_key: judgement
  ok_values: [OK]
  ng_values: [NG]
  confidence_key: confidence
  allow_code_fence: false           # true にすると ```json の囲みだけは外す
```

- 画像は「見本 → 検査対象 → 指示の文」の順に1つのメッセージで渡します。`layout: target_first` にすると「検査対象 → 指示 → `ref_note` の文 → 見本」の順になります（AWS サンプルの再現用）
- `inference` に書けるのは `temperature`・`maxTokens`・`topP` です。`topK` などは `additional_fields` に書きます
- AWS サンプルの応答（`objects` の各要素の `qc` が OK/NOK）は、`parse: {mode: objects, judgement_key: qc, ng_values: [NOK], confidence_key: null, defect_key: reason}` で読みます。空の配列なら OK、NOK が1つでもあれば NG です
- 品目ごとにプロンプトの一部を差し替えるときは、プロンプトに `{{item}}` のように書き、`prompt_vars: {pcb1: {item: "..."}, capsules: {item: "..."}}` を設定します
- 推論の設定などは `additional_fields` に書くと `additionalModelRequestFields` として渡します

### 2. 試運転してから本番

```bash
uv run vlmlab run configs/n1_B.yaml --dry-run      # 呼ばずに件数と単価の有無を確認
uv run vlmlab run configs/n1_B.yaml --limit 10     # 試運転（runs/n1_B__trial10/）
uv run vlmlab run configs/n1_B.yaml                # 本番（runs/n1_B/）。止まっても同じコマンドで続きから
```

- `pricing.yaml` で入力か出力の単価が null のモデルは、実行を断ります
- 次の呼び出しの前に「累計 + これまでで最も高かった1呼び出しの費用」が `budget_usd` を超えるなら止めます
- スロットリングなどは、指数バックオフで最大6回まで試します。それでも失敗した呼び出しは `api_error` を付けて記録し、次の実行で呼び直します

### 埋め込み＋k近傍法（条件 E）

```bash
uv run vlmlab embed configs/n1_E.yaml --dry-run     # 呼ばずに件数と単価を確認
uv run vlmlab embed configs/n1_E.yaml --limit 10    # 試運転：テスト画像10枚を埋め込むだけ
uv run vlmlab embed configs/n1_E.yaml               # 学習用240枚とテスト200枚を埋め込み、品目ごとに k近傍法で判定
```

- 埋め込みは `runs/<name>/embeddings.jsonl` に残すので、k を変えても呼び直しません
- 判定は `calls.jsonl`（ほかの条件と同じ形）に書くので、`vlmlab report` でそのまま集計できます。費用はテスト画像1枚の埋め込みの値で、学習用画像の費用は `summary.json` の `index_cost_usd` に分けて書きます
- 費用は画像1枚あたりの単価（`pricing.yaml` の `per_image_usd`）で数えます。応答ヘッダーの入力トークン数は、画像では 0 になるので費用には使いません

### 3. 集計

```bash
uv run vlmlab report runs/n1_A runs/n1_B runs/n1_C --out results --tag summary_200
```

`summary_200.csv`・`.md`（条件ごとの表）、`_by_category.csv`、`_business_metric.csv`、`_misclassified.csv` を出します。

## 記録の形（runs/<name>/calls.jsonl、1行1呼び出し）

| キー | 中身 |
| --- | --- |
| run_name, run_id, repeat | 実験名、実行 ID、何回目か |
| image_id, category, label, path | 画像 ID、品目、正解（normal / anomaly）、画像のパス |
| model_id, service_tier_requested, service_tier_resolved | モデル ID、頼んだ層、実際に処理された層（応答の serviceTier） |
| prompt_version, n_refs, image_long_edge | プロンプトの版、見本の枚数、画像の縮小 |
| raw_text | 応答の本文（そのまま） |
| parsed, parse_error | 読み取った JSON（読めなければ null）、読めなかった理由 |
| input_tokens, output_tokens, cache_read_tokens, cache_write_tokens | usage の値 |
| latency_s, server_latency_ms | 呼び出しの往復時間（秒）、応答の metrics.latencyMs |
| stop_reason | stopReason |
| cost_usd | トークン数 × 単価 |
| attempts, api_error | 試した回数、再試行しても失敗したときの内容 |
| timestamp | 呼び出した時刻（JST） |

使った設定とプロンプトの写しも `runs/<name>/` に残します。

## 評価指標

定義の詳細は [docs/contracts.md](docs/contracts.md) にあります。

- 「NG」を陽性として、適合率・再現率・F1、誤検知率（良品のうち NG とした割合）、見逃し率（不良品のうち OK とした割合）
- 繰り返しのある実験は、回ごとに計算して平均・最小・最大を出します
- JSON として読めなかった応答（解析失敗）は指標から除き、有効件数と解析失敗率を併記します
- 業務指標：見逃し率を 2%・5% 以下にしたときの良品の通過率。スコアは NG なら確信度、OK なら 1 − 確信度です。確信度が無い応答があれば計算しません
- 一致率（全回で同じ判定だった画像の割合。解析失敗の回があれば不一致）、レイテンシの p50・p95、1件あたり・1,000枚あたりの費用

## データ

[VisA](https://registry.opendata.aws/visa/)（Zou et al., ECCV 2022、CC BY 4.0）を使います。データ本体はコミットしていません。取得方法は [data/README.md](data/README.md) を見てください。
