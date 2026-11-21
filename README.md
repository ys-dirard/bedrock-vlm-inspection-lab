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
│   ├── cascade.py          段階判定（迷った画像だけ上位モデルに回す）を記録から計算する
│   ├── anomalydino.py      手元の AnomalyDINO で異常度を出す（AWS は使わない）
│   ├── prefilter.py        2段構成（AnomalyDINO → Nova）を記録から計算する。書き出しは prefilter_report.py
│   ├── cli.py              コマンドライン（vlmlab）
│   └── mmad/               MMAD の選択式の問題（取得・プロンプト・呼び出し・採点・集計）
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

## 手元の AnomalyDINO（12/10 の記事。AWS は使わない）

良品1枚の見本から、画像ごとの異常度を手元の PC で出します。公式実装（[dammsi/AnomalyDINO](https://github.com/dammsi/AnomalyDINO)、Apache-2.0）の `src/backbones.py`・`src/utils.py` を読み込み、DINOv2（Apache-2.0）の重みは torch.hub で取ります。PyTorch などは追加の依存（`anomalydino`）にしているので、ほかの記事の環境には入りません。

```bash
uv sync --extra anomalydino                                     # PyTorch（CPU 版）など。仮想環境は約1 GB になる
git clone https://github.com/dammsi/AnomalyDINO third_party/AnomalyDINO
git -C third_party/AnomalyDINO checkout dd35b0f1f6ff6146d196fe4efd3457a58ebcb1e0
uv run --extra anomalydino vlmlab anomalydino configs/ad1_anomalydino.yaml --dry-run
uv run --extra anomalydino vlmlab anomalydino configs/ad1_anomalydino.yaml --limit 8   # 試運転（runs/ad1_anomalydino__trial8/）
uv run --extra anomalydino vlmlab anomalydino configs/ad1_anomalydino.yaml             # 本番（200枚）
```

- `--extra anomalydino` を付けずに `uv sync` すると、PyTorch などは外れます
- 重み（約85 MB）と hub のコードは `data/torch_home/` に置きます（`--torch-home` で変えられる）
- 結果は `runs/<name>/scores.jsonl`（1枚1行：異常度、1枚あたりの時間、パッチの数）と `meta.jsonl`（環境、見本の特徴づくりの時間、メモリ）
- 計算の定義は [docs/contracts.md](docs/contracts.md) の CI-A1〜A5

### 2段構成（AnomalyDINO で絞り、上位だけ Nova 2 Lite に回す）

AnomalyDINO の異常度と、Nova 2 Lite の記録（11/11 の条件 B、`runs/n1_B`）を組み合わせて計算します。API は呼びません。

```bash
uv run --extra anomalydino vlmlab prefilter \
  --scores 448=runs/ad1_anomalydino 672=runs/ad1_anomalydino_r672 \
  --nova runs/n1_B --stage2 "Nova Pro=runs/n1_D" "Claude Sonnet 4.6=runs/n3_claude_refB" --out results
```

- 回す条件：異常度の上位 10・20・30・50%（品目ごと・全体）と、テスト画像の正解で決めた見逃し率 2%・5% のしきい値以上（楽観的な値）
- 回さなかった画像は OK とする場合と、AnomalyDINO の判定（見逃し率 2%・5% のしきい値）とする場合
- 回した画像の判定は Nova の repeat=1（費用も1回分）。3回の多数決は `maj_` の列（費用は3回分）
- `--stage2` は参考：2段目を1枚1回の別の記録に替えた表（`prefilter_summary.md` の「参考：2段目を替えた場合」）と `fig_prefilter_alt.png` を足す。解析失敗は回さない画像と同じ扱い（CI-F9・F10）
- `prefilter_summary.csv`・`.md`（表1）、`anomalydino_scores.csv`（画像ごと）、`anomalydino_score_summary.csv`（品目ごとの AUROC と分布）、`anomalydino_time.json`、`fig_prefilter.png` を出します。定義は CI-F1〜F8

## 学習データへの混入（12/22 の記事）

VisA の train の良品に、評価に使っていない不良品を 5% 混ぜ、Nova 2 Lite と手元の AnomalyDINO で見つけられるかを測ります。

```bash
uv run vlmlab noise splits                                              # data/splits/noise_train_420.csv（中立の ID、正解なし）と noise_truth_420.csv（正解）
uv run vlmlab run configs/nz1_nova2lite.yaml --limit 8                  # Nova の試運転（本番は --limit なしで420枚）
uv run --extra anomalydino vlmlab noise dino configs/nz1_anomalydino.yaml --limit 8   # AnomalyDINO の類似度の表の試運転
uv run --extra anomalydino vlmlab noise report --dino runs/nz1_anomalydino --nova runs/nz1_nova2lite --out results
```

- 判定に使う一覧には正解を書かず、正解は別のファイルに分けています。モデルに送るのは画像だけです
- 定義は [docs/contracts.md](docs/contracts.md) の CI-N1〜N7

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

## 段階判定（11/26 の記事）

1段目（Nova 2 Lite の3回、または埋め込みの距離）で「迷った」画像だけを上位モデル（Claude）に回したときの、F1・誤検知率・回した割合・1,000枚あたりの費用・レイテンシを計算します。上位モデルは全件で1回ずつ呼んでおき、回す条件ごとの結果は記録から組み合わせるので、条件を何通り試しても API は呼びません。

```bash
uv run vlmlab run configs/n3_claude_refB.yaml --limit 10      # 試運転（Claude Sonnet 4.6、JP Geo）
uv run vlmlab run configs/n3_claude_refB.yaml                 # 本番（200枚×1回）
uv run vlmlab cascade --nova runs/n1_B --claude runs/n3_claude_refB     --embeddings runs/n1_E/embeddings.jsonl --out results
```

- 回す条件は R1（3回の判定が割れた）、R2（確信度<0.8）、R3（NG と言った）、R4（R1 または R3）、R5（確信度<τ）、R6（同じ品目の学習用の良品との最小コサイン距離が、良品どうしの距離の百分位以上）です。定義は [docs/contracts.md](docs/contracts.md) の CI-K1〜K12 にあります
- `cascade_rules.csv`・`.md`（条件ごとの表）、`fixed_and_broken.csv`、`cascade_images.csv`、`confidence_bands.csv`、`business_metric.csv`、`e_thresholds.csv` を出します
- Claude は説明のあとに ```` ```json ```` の囲みで JSON を付けるので、設定で `parse.allow_trailing_code_fence: true` にしています（CI-P7）

## MMAD の選択式の問題（11/19 の記事）

[MMAD](https://github.com/jam-cc/MMAD)（ICLR 2025）の選択式の問題を、Bedrock の VLM に解かせて正答率を出します。プロンプト・画像の縮小・回答の読み取り・集計は、公式の評価スクリプト（commit `49db95b` の `evaluation/examples/GPT4/gpt4o.py` と `helper/summary.py`）に合わせています。違うところ（temperature 0、maxTokens 1,024 など）は記事に書いています。

```bash
uv run vlmlab mmad prepare                                  # mmad.json を取り（SHA-256 を確認）、使う画像だけを Hugging Face の ZIP から部分的に取り出す（約2.2 GB）
uv run vlmlab mmad run configs/m1_nova2lite.yaml --limit 10 # 試運転
uv run vlmlab mmad run configs/m1_nova2lite.yaml            # 本番
uv run vlmlab mmad report runs/m1_nova2lite runs/m1_haiku45_ondemand --out results
uv run vlmlab mmad report runs/... --out results644 --subset data/splits/mmad_644.csv   # 同じ問題にそろえて集計
```

- 使った問題の一覧は `data/splits/mmad_1050.csv`（7課題×150問、random_state=42）と `mmad_644.csv`（そこから各課題92問。record_id は共通）
- 設定は `configs/m1_*.yaml`。Claude は Messages API の本文（InvokeModel）、Nova 2 Lite は Converse で呼びます
- `vlmlab mmad batch-prepare` / `batch-collect` はバッチ推論用ですが、アカウントでバッチ推論が許可されておらず、ジョブでは試せていません
- 問題と画像はコミットしていません。MMAD は研究目的での利用とされています。画像の元のデータセット（MVTec AD・MVTec LOCO・VisA・GoodsAD など）のライセンスにも従ってください
