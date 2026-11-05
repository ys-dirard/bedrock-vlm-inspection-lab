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
