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

## MVTec AD・MVTec AD 2・MVTec LOCO AD（12/7・12/15 の記事）

- 配布: MVTec の公式ページのフォーム（氏名・メール・非商用の確認のチェックが必須）。このリポジトリでは取得しない
  - MVTec AD: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
  - MVTec AD 2: https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
  - MVTec LOCO AD: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad
- ライセンス: どれも CC BY-NC-SA 4.0（非商用）。画像の転載は最小限にする
- 置き場所: `data/mvtec/` の下にデータセットごとに展開する（コミットしない）

```
data/mvtec/
├── mvtec_ad/<品目>/{train/good, test/<good|欠陥の種類>, ground_truth}
├── mvtec_ad_2/<品目>/{train/good, validation/good, test_public/{good,bad,ground_truth}, test_private, test_private_mixed}
└── mvtec_loco_ad/<品目>/{train/good, validation/good, test/{good,logical_anomalies,structural_anomalies}, ground_truth}
```

アーカイブを展開したときの最上位のディレクトリ名がこれと違う場合は、`--root` で品目の1つ上を指してください。品目ごとの枚数を論文の表と照合し、合わなければ止まります（docs/contracts.md の CI-V2）。

```bash
# 12/7（品目は記事の書き手が決める。例）
uv run vlmlab mvtec-splits mvtec_ad   --root data/mvtec/mvtec_ad   --categories hazelnut carpet --name mvtec_ad_100
uv run vlmlab mvtec-splits mvtec_ad_2 --root data/mvtec/mvtec_ad_2 --categories walnuts fabric --name mvtec_ad2_100
# 12/15
uv run vlmlab mvtec-splits mvtec_loco_ad --root data/mvtec/mvtec_loco_ad --categories breakfast_box screw_bag --name mvtec_loco_200
# 実行は画像の根を data/mvtec にする
uv run vlmlab run configs/ac07_ad_nova.yaml --limit 10 --image-root data/mvtec
```

| ファイル | 中身 |
| --- | --- |
| `mvtec_ad_100.csv`・`mvtec_ad_100_refs.csv` | MVTec AD の2品目 × test の良品25枚・不良品25枚（欠陥の種類はまとめて選ぶ）と、train の良品から各3枚の見本 |
| `mvtec_ad2_100.csv`・`mvtec_ad2_100_refs.csv` | MVTec AD 2 の2品目 × test_public の good 25枚・bad 25枚と見本。test_private は正解が非公開なので使わない |
| `mvtec_loco_200.csv`・`mvtec_loco_200_refs.csv` | MVTec LOCO AD の2品目 × test の good 50枚・logical_anomalies 25枚・structural_anomalies 25枚と見本 |

列は `image_id, category, label, split, path, dataset, subtype`（`path` は `data/mvtec` からの相対パス）。抽出は image_id の順に並べてから `DataFrame.sample(n, random_state=42)` です。
