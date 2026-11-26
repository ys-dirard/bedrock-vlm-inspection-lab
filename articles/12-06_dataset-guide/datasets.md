# 公開データセットのライセンスと入手方法（確認日 2026-11-16）

- 元の表は `datasets.csv`（UTF-8 BOM つき）。ここでは同じ内容を短くした表にし、原文と下書きとの照合を足す
- 値は公式のページ（提供元のサイト、Hugging Face、GitHub、Registry of Open Data on AWS）から取った。論文の値は「二次情報」と書いた
- AWS の API・CLI は呼んでいない（S3 の一覧の可否は Registry の記載だけ）
- 開いた URL の一覧は `sources.md`

## 表

| データセット | 提供元 | 規模 | ライセンス | 商用 | 再配布・派生物 | 表示・引用 | 入手方法 | 登録 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VisA | Amazon（amazon-science、Registry は AWS が管理） | 10,821枚（正常9,621・異常1,200）、12品目 | CC BY 4.0 | 可 | 可（CC BY 4.0。独自の記述なし） | 要（論文の引用を依頼。Registry に引用の書式） | S3 `amazon-visual-anomaly`（us-west-2）の `VisA_20220922.tar`、`--no-sign-request`。HTTPS の直接ダウンロードも | 不要 |
| MVTec AD | MVTec Software GmbH | 5,000枚以上、15カテゴリ（二次情報：5,354枚＝学習・検証3,629、テスト1,725） | CC BY-NC-SA 4.0 | 不可（迷うときは問い合わせ） | 独自の記述なし（非商用・継承の範囲で可） | ページに引用の依頼文はない（論文へのリンクのみ） | 公式ページのフォーム | 要（フォーム。項目は【要確認】） |
| MVTec AD 2 | MVTec Software GmbH | 8,000枚以上、8シナリオ | CC BY-NC-SA 4.0 | 不可（迷うときは問い合わせ） | 独自の記述なし（非商用・継承の範囲で可）。private テストの正解は非公開 | 要（科学的な仕事で使うなら論文を引用） | 公式ページのフォーム。private テストは評価サーバー | 要（フォーム。項目は【要確認】） |
| MVTec LOCO AD | MVTec Software GmbH | 3,644枚、5カテゴリ | CC BY-NC-SA 4.0 | 不可（迷うときは問い合わせ） | 独自の記述なし（非商用・継承の範囲で可） | 要（科学的な仕事で使うなら論文を引用） | 公式ページのフォーム | 要（フォーム。項目は【要確認】） |
| MMAD | 論文の著者（ICLR 2025） | 画像8,366枚。問題は GitHub・論文で39,672問、HF の現行版（2026-10-22 修正）で39,670問 | 明記なし（HF「research use only」、元データの条件に従う） | MMAD としては「research use only」。画像は元データの条件（MVTec 由来は非商用） | 明記なし（HF で元データの画像を ZIP で配布） | 要（論文の引用を依頼） | Hugging Face（ゲートなし）。GitHub に評価コード | 不要 |
| Real-IAD | Tencent Youtu Lab ほか | 公式ページに枚数なし。30品目。二次情報：151,050枚（正常99,721・異常51,329）、カメラ5台 | HF のメタデータは CC BY-NC-SA 4.0、本文は「research purpose only」。GitHub の LICENSE は Apache 2.0（コード） | 不可と読める | 明記なし（LICENSE.txt は【要確認】） | BibTeX の掲示のみ | Hugging Face（ゲートつき・自動承認） | 要（HF ログインと情報の共有に同意） |
| ARMBench | Amazon（Amazon Robotics） | 欠陥検出：欠陥画像13,303枚（タスクページ）／19,000+枚（ホーム）、正常100,000枚、欠陥動画4,075本。ほかに領域分割50,000+枚、同定200,000+物体 | CC BY 4.0 | 可 | 可（CC BY 4.0） | 要（出版物で使うなら引用） | Hugging Face `amazon/armbench-*`（6リポジトリ、ゲートなし）、または Data Request Form | 不要（フォームは任意） |

ライセンスの URL：CC BY 4.0 は https://creativecommons.org/licenses/by/4.0/ 、CC BY-NC-SA 4.0 は https://creativecommons.org/licenses/by-nc-sa/4.0/ 。

## 原文の抜き書き

### VisA

- 「Largest Visual Anomaly detection dataset containing objects from 12 classes in 3 domains across 10,821(9,621 normal and 1,200 anomaly) images.」 — https://registry.opendata.aws/visa/
- 「Access (No AWS account required)」「aws s3 ls --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar/」 — https://registry.opendata.aws/visa/
- License 欄は URL のみ「https://creativecommons.org/licenses/by/4.0/」 — https://registry.opendata.aws/visa/
- 「Visual Anomaly (VisA) was accessed on DATE from https://registry.opendata.aws/visa.」 — https://registry.opendata.aws/visa/
- 「There are 10,821 images with 9,621 normal and 1,200 anomalous samples.」 — https://github.com/amazon-science/spot-diff
- 「The data is released under the CC BY 4.0 license.」 — https://github.com/amazon-science/spot-diff
- 「We host the VisA dataset in AWS S3 and you can download it by this URL」（リンク先 https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar ） — https://github.com/amazon-science/spot-diff
- 「Please cite the following paper if this dataset helps your project:」 — https://github.com/amazon-science/spot-diff
- LICENSE-DATASET の冒頭「Attribution 4.0 International」 — https://raw.githubusercontent.com/amazon-science/spot-diff/main/LICENSE-DATASET

### MVTec AD・MVTec AD 2・MVTec LOCO AD（License Terms は3ページとも同じ文）

- 「The data is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).」
- 「In particular, it is not allowed to use the dataset for commercial purposes.」
- 「If you are unsure whether or not your application violates the non-commercial use clause of the license, please contact us.」
- 「If you have any questions or comments about the dataset, feel free to contact us via email.」
- 出典：https://www.mvtec.com/company/research/datasets/mvtec-ad 、https://www.mvtec.com/company/research/datasets/mvtec-ad-2 、https://www.mvtec.com/company/research/datasets/mvtec-loco

各ページの規模：

- 「It contains over 5000 high-resolution images divided into fifteen different object and texture categories.」 — https://www.mvtec.com/company/research/datasets/mvtec-ad
- 「It expands existing benchmarks by eight new anomaly detection scenarios with more than 8,000 high-resolution images in total.」 — https://www.mvtec.com/company/research/datasets/mvtec-ad-2
- 「...captured under various lighting conditions, which are not necessarily contained in the training data.」「for the second part ground truth information is non-public.」 — https://www.mvtec.com/company/research/datasets/mvtec-ad-2
- 「It contains 3644 images from five different categories inspired by real-world industrial inspection scenarios.」 — https://www.mvtec.com/company/research/datasets/mvtec-loco
- 「If you use this dataset in scientific work, please cite our paper:」 — MVTec AD 2 と MVTec LOCO AD のページ（MVTec AD のページにはこの文がない）
- 入手：3ページとも「Download dataset（Download Dataset）」「Fill in the form」（フォームは software.mvtec.com の埋め込みスクリプトで読み込まれる）

二次情報：

- 「5354 high-resolution color images」「3629 images for training and validation and 1725 images for testing」 — MVTec AD の CVPR 2019 論文 https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf

### MMAD

- 「Please note that these datasets are for research use only and should be used in accordance with their respective copyright and licensing agreements.」 — https://huggingface.co/datasets/jiang-cc/MMAD
- 「For research purposes, we have provided a ZIP file containing all the datasets」 — https://huggingface.co/datasets/jiang-cc/MMAD
- 「`random_templates` and the total of 8,366 query images are unchanged. This revision also synchronizes ten existing question corrections from the GitHub repository, resulting in 39,670 question records」（Annotation corrections (2026-10-22)） — https://huggingface.co/datasets/jiang-cc/MMAD
- 「If you use MMAD in your research, please cite the MMAD paper below.」 — https://huggingface.co/datasets/jiang-cc/MMAD 、https://github.com/jam-cc/MMAD
- 「We collected 8,366 samples from 38 classes of industrial products across 4 public datasets, generating a total of 39,672 multiple-choice questions in 7 key subtasks.」 — https://github.com/jam-cc/MMAD
- HF カードに `license:` の項目なし（README.md に YAML の前置きがない）。GitHub API の `"license": null`、リポジトリ直下に LICENSE ファイルなし — https://huggingface.co/datasets/jiang-cc/MMAD 、https://api.github.com/repos/jam-cc/MMAD
- 元データ：DS-MVTec の HF カード `license: mit` — https://huggingface.co/datasets/DefectSpectrum/Defect_Spectrum 。GoodsAD のリポジトリは GPL-3.0（GitHub API の spdx_id）で、README にデータのライセンスの記載なし — https://github.com/jianzhang96/GoodsAD

### Real-IAD

- 「Real-IAD is released for research purpose only.」「Your access request will be automatically apporoved.」（原文のつづりのまま） — https://huggingface.co/datasets/Real-IAD/Real-IAD
- 「You need to agree to share your Hugging Face information to access this dataset」、License: cc-by-nc-sa-4.0 — https://huggingface.co/datasets/Real-IAD/Real-IAD
- 「If you are interested in using the dataset, you can download it at Hugging Face . Your access request will be automatically approved.」 — https://realiad4ad.github.io/Real-IAD/
- 「This website is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.」（ウェブサイトのライセンス） — https://realiad4ad.github.io/Real-IAD/
- 「Real-IAD is licensed under the Apache License Version 2.0 except for the third-party components listed below.」 — https://github.com/Tencent/AnomalyDetection_Real-IAD （LICENSE）
- 「`audiojack` is one of the 30 objects in Real-IAD」 — https://github.com/Tencent/AnomalyDetection_Real-IAD （README）
- 二次情報：「150K high-resolution images of 30 different objects」 — https://arxiv.org/abs/2403.12580 。「There are a total of five cameras capturing the object from different angles」、表の「30 99,721 51,329 151,050」 — https://arxiv.org/html/2403.12580

### ARMBench

- 「These datasets are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). Use the following citations if you use these datasets for publication.」 — https://www.armbench.com/
- 「19,000+ images and 4,000+ videos for rare, but costly, robot-induced defects such as multi-pick and packaging defects. Dataset contains 100,000+ no defect cases.」 — https://www.armbench.com/
- 「This dataset comprises 13,303 images of objects with defects」「100,000 images of objects with no defects are available in the dataset.」「This dataset comprises 4,075 videos of objects with defects.」 — https://www.armbench.com/defects.html
- 「All six ARMBench datasets are now published on HuggingFace under the official Amazon organization」「The Data Request Form below remains available if you would rather receive download links by email.」 — https://www.armbench.com/data.html
- 「The datasets are collected in Amazon warehouses」 — https://www.armbench.com/
- 「Further restrictions | None are applied. This redistribution adds no terms beyond CC BY 4.0.」「the site's two pages disagree with each other」 — https://huggingface.co/datasets/amazon/armbench-defect-image

### Creative Commons のコモンズ証（参考）

- CC BY 4.0：「copy and redistribute the material in any medium or format for any purpose, even commercially.」 — https://creativecommons.org/licenses/by/4.0/
- CC BY-NC-SA 4.0：「NonCommercial — You may not use the material for commercial purposes.」「ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.」 — https://creativecommons.org/licenses/by-nc-sa/4.0/

## 下書きとの照合

| 下書きの記述 | 判定 | 公式の値・メモ |
| --- | --- | --- |
| VisA 10,821枚（正常9,621・異常1,200） | 一致 | Registry・GitHub とも同じ |
| VisA 12品目 | 一致 | 「12 classes in 3 domains」「12 subsets」 |
| VisA CC BY 4.0（商用可） | 一致 | GitHub README・LICENSE-DATASET。Registry は URL のみ |
| VisA は S3 で公開、認証なしで取得できる | 一致 | 「No AWS account required」 |
| `aws s3 ls --no-sign-request s3://amazon-visual-anomaly/` | 違う（一部） | バケット名と `--no-sign-request` は一致。Registry の記載は `s3://amazon-visual-anomaly/VisA_20220922.tar/`。バケット直下を一覧できるかは未確認 |
| VisA と ARMBench はどちらも Amazon が公開 | 一致 | VisA は amazon-science・Registry は AWS が管理。ARMBench は © Amazon.com、HF は amazon の organization |
| MVTec AD 15品目 | 一致 | 「fifteen different object and texture categories」（「品目」より「カテゴリ」が原文に近い） |
| MVTec AD 規模【要確認】 | 未確認→値あり | 公式「over 5000」。二次情報で5,354枚 |
| MVTec AD CC BY-NC-SA 4.0 | 一致 | |
| MVTec AD 2 8,000枚以上 | 一致 | 「more than 8,000」 |
| MVTec AD 2 CC BY-NC-SA 4.0 | 一致 | |
| MVTec AD 2 の中身「透明な物体・照明の変化・微小な欠陥」 | 一部未確認 | 照明は「various lighting conditions」で一致。透明な物体・微小な欠陥は公式ページの本文になし |
| MVTec LOCO AD のライセンス【要確認：CC BY-NC-SA 4.0】 | 一致 | 【要確認】を外せる |
| MVTec LOCO AD 規模【要確認】 | 未確認→値あり | 3,644枚・5カテゴリ |
| MVTec の公式ページに「迷ったら問い合わせる」 | 一致 | 「If you are unsure whether or not your application violates the non-commercial use clause of the license, please contact us.」（3ページとも） |
| MMAD 画像8,366枚 | 一致 | |
| MMAD 問題39,672問 | 違う（版による） | GitHub README・論文は39,672。HF の現行版（2026-10-22 の修正）は39,670、mmad.json を数えても39,670 |
| MMAD CC BY-NC-SA 4.0（非商用） | 違う | 公式のページにライセンスの明記なし。HF「research use only」「in accordance with their respective copyright and licensing agreements」。MVTec 由来の画像は CC BY-NC-SA 4.0、VisA 由来は CC BY 4.0 |
| Real-IAD 30品目 | 一致 | GitHub README |
| Real-IAD 5方向 | 未確認（二次情報では一致） | 公式のページに記載なし。論文に「a total of five cameras」 |
| Real-IAD 約15万枚 | 未確認（二次情報では一致） | 公式のページに枚数なし。論文で150K（表は151,050） |
| Real-IAD【要確認：配布サイトの条件】 | 値あり | HF のゲート（自動承認）、「research purpose only」、HF メタデータ CC BY-NC-SA 4.0。GitHub の LICENSE は Apache 2.0（コード） |
| ARMBench 規模【要確認】 | 値あり（ページ間で不一致） | 欠陥画像13,303枚（タスクページ）／19,000+枚（ホーム）、正常100,000枚 |
| ARMBench【要確認：CC BY 4.0】 | 一致 | armbench.com と HF のカード |
| ARMBench は Amazon の倉庫のピッキング画像（欠陥・商品の同定） | 一致 | ほかに領域分割と Vulcan（Stow）のタスクもある |
| Appendix の URL（VisA の Registry、MVTec AD 2、MMAD の HF） | 一致 | 3つとも開けた |

## 【要確認】一覧

1. MVTec AD・MVTec AD 2・MVTec LOCO AD のダウンロードフォームの項目と、送信後の受け取り方（フォームは JavaScript で読み込まれ、取得した HTML に項目がない）
2. VisA：`aws s3 ls --no-sign-request s3://amazon-visual-anomaly/`（バケット直下）で一覧できるか（AWS を呼ばない方針のため未確認。Registry の記載は tar のパスまで）
3. Real-IAD：HF の `LICENSE.txt` とカード全文（ゲートの内側。ログインして同意しないと読めない）
4. Real-IAD：公式のページでの枚数と撮影方向の数（統計は画像だけで、今は論文（二次情報）の値しかない）
5. MVTec AD 2：「透明な物体」「微小な欠陥」の記述の出典（公式ページの本文にない。論文で確かめる）、分割ごとの枚数
6. MVTec LOCO AD：構造的な異常と論理的な異常の内訳（公式ページに記載なし）
7. ARMBench：欠陥画像の枚数（13,303 と 19,000+ のどちらを書くか。公式のページの間で不一致）
8. MMAD の元データ GoodsAD のデータのライセンス（リポジトリは GPL-3.0、データの条件は明記なし）
