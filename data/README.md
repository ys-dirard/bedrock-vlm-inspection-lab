# データ

データ本体はコミットしません。`data/splits/` の画像 ID の一覧だけをコミットしています。

## VisA

- 出典: Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022
- 配布: [Registry of Open Data on AWS](https://registry.opendata.aws/visa/)（`s3://amazon-visual-anomaly/VisA_20220922.tar`、us-west-2、認証なしで取得可）
- ライセンス: CC BY 4.0
- 分割: tar に含まれる `split_csv/1cls.csv`（[spot-diff](https://github.com/amazon-science/spot-diff) リポジトリと同じもの）

## 取得

```bash
uv run vlmlab fetch      # data/raw/VisA_20220922.tar（約1.9 GB）を取得し、data/visa/ に展開する
uv run vlmlab splits     # data/splits/ の CSV を作り直す（random_state=42 なので同じ内容になる）
```

AWS CLI を使う場合は次と同じです。

```bash
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar data/raw/
mkdir -p data/visa && tar -xf data/raw/VisA_20220922.tar -C data/visa
```

## splits/

| ファイル | 中身 |
| --- | --- |
| `visa_test_200.csv` | 基本のテスト。pcb1 と capsules の test から、良品50枚・不良品50枚ずつ（計200枚） |
| `visa_refs.csv` | 良品の見本。12品目それぞれの train の良品から3枚（`ref_rank` 1〜3。テストとは重ならない） |
| `visa_wide_120.csv` | 品目の広がりを見るとき。12品目 × test の良品5枚・不良品5枚（計120枚） |

列は `image_id, category, label, split, path`（`path` は VisA の根からの相対パス）です。抽出はすべて `pandas.DataFrame.sample(n, random_state=42)` です。

## ARMBench（12/19 の記事）

- 出典: ARMBench（Amazon、CC BY 4.0、http://armbench.com 、arXiv:2303.16382）。配布は Hugging Face の `amazon/armbench-defect-image`（ゲートなし）
- 本体 `archive/armbench-defects-image-0.1.tar.gz` は 70,571,002,037 バイト（65.7 GiB）。gzip なので一部だけは取り出せない
- 試運転では、同じ所の見本 `archive/armbench-defects-tiny.tar.gz`（69,609,824 バイト、52枚）と公式の分割 `annotations/train.csv`・`test.csv` を使った

```bash
mkdir -p data/armbench/raw data/armbench/tiny
B=https://huggingface.co/datasets/amazon/armbench-defect-image/resolve/main
curl -L -o data/armbench/raw/armbench-defects-tiny.tar.gz $B/archive/armbench-defects-tiny.tar.gz
curl -L -o data/armbench/raw/train.csv $B/annotations/train.csv
curl -L -o data/armbench/raw/test.csv  $B/annotations/test.csv
tar xzf data/armbench/raw/armbench-defects-tiny.tar.gz -C data/armbench/tiny
uv run vlmlab armbench splits --root data/armbench/tiny/defects-image-tiny --splits train test none \
  --n-normal 3 --n-defect 6 --out data/splits/armbench_tiny_trial.csv
uv run vlmlab run configs/ab0_tiny_nova2lite.yaml --visa-root data/armbench/tiny/defects-image-tiny
```

- `data/armbench/` は .gitignore に入れている
