# 記事ごとの費用の実績（12/11 の記事、2026-11-21 集計）

- 値はすべて、各記事の `results/cost_total.md`（記録のトークン数 × 単価）を写したもの。写した行は `cost_by_article.csv`（出典のパスは `source` 列）
- 集計: `src/aggregate_costs.py`（記事ごとの行の合計が各 `cost_total.md` の合計と合うことを確かめてから出力。単体テストは `src/test_aggregate_costs.py`、9件すべて合格）
- 出典の場所: 11/11・11/19・11/26 は main（`november/*/results/cost_total.md`）。11/30 と12月の記事は、各記事のワークツリー（`.worktrees/<名前>/…/results/cost_total.md`）。ほかのワークツリーは読んだだけ
- 「1,000枚あたり」は本番の行だけ。費用 ÷ のべ判定数 × 1,000（例: 200枚×3回なら 600 で割る）。11/19 は「1,000問あたり」（MMAD の問題）、11/30 はモデル以外（AgentCore・Step Functions など）も含む
- この集計で AWS の API は呼んでいない（費用 0 USD）

## 記事ごと

| 公開日 | 記事 | 実行日（JST） | 主なモデル | 呼び出し | 試運転など（USD） | 本番（USD） | 合計（USD） | 本番の1,000枚あたり（USD） | 出典 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11/11 | aws-zero-training-visa | 11/05 | Nova 2 Lite、Nova Pro、埋め込み | 2,684 | 0.0852 | 1.7798 | **1.8650** | 0.060（埋め込み）〜5.196（Nova Pro） | main `november/2026-11-11_aws-zero-training-visa/results/cost_total.md` |
| 11/19 | mmad-remeasure | 11/05 | Haiku 4.5、Nova 2 Lite、Sonnet 4.6 | 3,854 | 0.1314 | 5.5274 | **5.6588** | 0.293（Nova、0-shot）〜4.577（Sonnet）※問題あたり | main `november/2026-11-19_mmad-remeasure/results/cost_total.md` |
| 11/26 | nova-claude-cascade | 11/11 | Sonnet 4.6 | 210 | 0.0711 | 1.4855 | **1.5566** | 7.428 | main `november/2026-11-26_nova-claude-cascade/results/cost_total.md` |
| 11/30 | serverless-inspection-platform | 11/20 | Nova 2 Lite、Qwen3 VL、Sonnet 4.6＋AgentCore など | ― | 0.706 | 2.675 | **約 3.38** | 1.184（Qwen）〜39.207（Claude） | `.worktrees/platform-2026-11-30/november/2026-11-30_serverless-inspection-platform/results/cost_total.md`、`cost_breakdown_per1000.csv` |
| 12/1 | eval-yardstick | ― | （呼んでいない） | 0 | ― | ― | **0** | ― | `.worktrees/yardstick-2026-12-01/…/results/yardstick_note.md` |
| 12/2 | defect-localization | 11/16 | Nova 2 Lite | 2,670 | 0.1508 | 0.7690 | **0.9198** | 0.327（A）〜1.130（C、4回で1枚） | `.worktrees/localization-2026-12-02/…/results/cost_total.md` |
| 12/3 | defect-fewshot | 11/16 | Nova 2 Lite | 1,475 | 0.0331 | 0.6307 | **0.6638** | 0.235（キャッシュ）〜0.640（見本3枚） | `.worktrees/fewshot-2026-12-03/…/results/cost_total.md` |
| 12/4 | reinvent-1_vlm-cheatsheet | 11/16 | 16モデル（確認だけ） | 63 | 0.0370 | ― | **0.0370** | ― | `.worktrees/cheatsheet-2026-12-04/…/results/cost_total.md` |
| 12/5 | image-input-cost | 11/16 | Haiku 4.5、Nova 2 Lite | 840 | 0.0686 | 1.3580 | **1.4266** | 0.623（Nova）〜3.705（Haiku、長辺1568） | `.worktrees/imagecost-2026-12-05/…/results/cost_total.md` |
| 12/6 | reinvent-2_dataset-guide | ― | （呼んでいない） | 0 | ― | ― | **0** | ― | `.worktrees/datasets-2026-12-06/…/results/deviations.md` |
| 12/7 | mvtec-contamination | ― | ― | ― | ― | ― | **記録なし** | ― | 2026-11-21 時点で results/ が空 |
| 12/8 | flex-batch | 11/16・11/21 | Nova 2 Lite（Standard・Flex） | 820 | 0.0050 | 0.1987 | **0.2036** | 0.165（Flex）〜0.331（Standard） | `.worktrees/flex-2026-12-08/…/results/cost_total.md` |
| 12/9 | reasoning-effort | 11/16 | Nova 2 Lite（推論あり） | 645 | 0.0889 | 1.3353 | **1.4242** | 1.537（low）〜15.920（high） | `.worktrees/reasoning-2026-12-09/…/results/cost_total.md` |
| 12/10 | anomalydino-prefilter | ― | （呼んでいない。手元の CPU） | 0 | ― | ― | **0** | ― | `.worktrees/dino-2026-12-10/…/results/deviations.md` |
| 12/12 | confidence-calibration | 11/16 | Nova 2 Lite | 1,052 | 0.0166 | 0.3419 | **0.3585** | 0.342（200枚×5回） | `.worktrees/calib-2026-12-12/…/results/cost_total.md` |

- 試運転・本番の小計と合計は、各 `cost_total.md` に書かれた値（11/19 と 12/9 は、4桁に丸めた行を足すと 0.0001 ずれる）
- 11/30 は「2026-11-20 時点」の値で、EventBridge（単価未確認）と置いておくだけの費用（11/21 以降）がまだ入っていない
- 12/8 の Flex の夜の回は 11/16 に73件（0.012109 USD）、11/21 に続きの127件（0.021000 USD）。`raw/ac08_flex__night1.jsonl` の `timestamp` の日付で分けた

## まとめた数字

| 範囲 | 合計（USD） | 内訳 |
| --- | --- | --- |
| 11月の4本（11/11・11/19・11/26・11/30） | **12.46** | 1.8650 ＋ 5.6588 ＋ 1.5566 ＋ 約3.38 |
| 11月の3本（11/30 を除く） | 9.0804 | 11/26 の `cost_total.md` の表と同じ |
| 12/1〜12/10（12/7 は記録なし） | **4.6751** | 0.9198 ＋ 0.6638 ＋ 0.0370 ＋ 1.4266 ＋ 0.2036 ＋ 1.4242（12/1・12/6・12/10 は 0） |
| 12/10 までの累計（11月の4本＋12/1〜12/10） | **17.14** | 12.46 ＋ 4.6751 |
| 参考: 12/12（検証済み） | 0.3585 | |
| 記録のあるすべて | 17.49 | |

- どの記事も1記事の上限 6 USD に収まっている。最も高いのは 11/19（5.6588 USD、上限の94%）
- draft.md の【結果：12/10 までの累計】に入れる範囲（11/30 を含めるか、12/12 を含めるか）は書き手が決める

## モデルごと（記録のあるすべての記事）

| モデル | 費用（USD） | 呼び出し（分かる分） |
| --- | --- | --- |
| `jp.amazon.nova-2-lite-v1:0`（Standard） | 5.1726 | 10,602 |
| `jp.anthropic.claude-sonnet-4-6` | 4.5994 | 874 |
| 11/30 の構成（モデル＋AgentCore・Step Functions など。モデル別には分けていない） | 3.3807 | 295 |
| `jp.anthropic.claude-haiku-4-5-20251001-v1:0` | 3.1359 | 1,700 |
| `apac.amazon.nova-pro-v1:0` | 1.0895 | 210 |
| `jp.amazon.nova-2-lite-v1:0`（Flex） | 0.0679 | 410 |
| `amazon.nova-2-multimodal-embeddings-v1:0` | 0.0272 | 454 |
| 12/4 だけで呼んだ12モデル（Kimi・Mistral・Gemma・Qwen・Nemotron・Nova Lite） | 0.0215 | （モデルごとの回数は記録に無い） |

- 呼び出しの回数では Nova 2 Lite が全体の約74%（10,602 ÷ 14,250。回数の分からない 12/4 と、モデル別に分けていない 11/30 を除く）だが、費用では Claude（Sonnet 4.6・Haiku 4.5）の合計 7.7353 USD が Nova 2 Lite の 5.1726 USD より多い
- 本番の1,000枚あたりの比: Sonnet 4.6（11/26）7.428 USD ÷ Nova 2 Lite（11/11 の n1_B）0.3315 USD ＝ 約22倍。画像の入力だけなら、12/4 の `image_cost_per1000.csv` で Nova 2 Lite 0.092、Sonnet 4.6 は原寸 5.062（約55倍）・長辺768px 1.954（約21倍）USD / 1,000枚
- 推論（12/9）: 同じ Nova 2 Lite でも、high は 15.920 USD / 1,000枚で、推論なし（11/11 の n1_B、0.3315）の約48倍

## Cost Explorer の「11月の累計 20.37 USD（11/20 時点）」の出どころと、記録との差

### 出どころ

- 11/30 の記事の事実の確認（セッション0）の `results/facts.md` の「予算について（参考。2026-11-20 に Cost Explorer で確認）」（`.worktrees/platform-2026-11-30/november/2026-11-30_serverless-inspection-platform/results/facts.md`）。plan.md にも「11/20 の Cost Explorer で 11/1〜11/20 の累計が 20.37 USD（税込み）」とある
- 書かれている内訳: 税 1.86、Amazon Bedrock 6.36、Claude Haiku 4.5 6.32、Claude Sonnet 4.6 4.60、VPC（使っていない公開 IPv4 アドレス）1.09 USD（小数2桁）。この確認の生の結果（API の応答）は記事フォルダに残っていない

### 記録との突き合わせ（分かる範囲）

11/30 の本番（11/20 の昼）は、セッション0の確認のあとなので入っていないと見ている（推定）。また Cost Explorer の Amazon Bedrock の値が、11/16（UTC）までの記録の合計と合うので、20.37 USD には 11/16 までの使用が入っている、と判断した。

| Cost Explorer の行（11/20 時点） | 金額（USD） | 記録（実行日 11/5・11/11・11/16 の記事の分） | 差 | 差の説明 |
| --- | --- | --- | --- | --- |
| Amazon Bedrock（Nova など） | 6.36 | 6.3576 | ほぼ 0 | 日ごとの照合（11/10・11/15 の `cost_explorer.md`、11/21 の `billing_check.md`）で合っている。差は記録に残らなかった2回分（12/2 と 12/8、計 約0.0005 USD） |
| Claude Sonnet 4.6 | 4.60 | 4.5994 | ほぼ 0 | 11/19・11/26・12/4 の記録と合う |
| Claude Haiku 4.5 | 6.32 | 3.1359 | **＋3.18** | うち 2.4650 は 2026-11-05 の分で、同じ日に動いた 11/19 の LT 用のエージェント（記事の記録の外）と見ている（11/19 の `cost_explorer.md`：Cost Explorer 4.3022 − 記録 1.8372）。残りの約 0.72 は、記事の記録とは結び付けられない（11/5・11/15 以外の日の分。何の分かは未確認） |
| VPC（使っていない公開 IPv4 アドレス） | 1.09 | 0 | ＋1.09 | 記事の検証とは関係しない。11/14 以降は日別 0（facts.md） |
| そのほか | 0.14 | 0 | ＋0.14 | 20.37 から上の行と税を引いた残り。Cost Explorer の API（記録にあるのは 11/10 の2回・11/15・11/16 の各1回で計 0.04 USD）、S3（11/5、0.0004 USD）、AgentCore（11/5、LT の分）などが入っていると見ている。内訳は未確認 |
| 税 | 1.86 | ― | ＋1.86 | 記録は税抜き |
| **合計** | **20.37** | **14.0929** | **＋6.28** | 記事の記録は税抜きで 14.09 USD。Cost Explorer の税抜きは 18.51 USD |

- 記録の 14.0929 USD は、実行日（JST）が 11/5・11/11・11/16 の行の合計（11/11・11/19・11/26・12/2・12/3・12/4・12/5・12/8 の 11/21 の127件を除く分・12/9・12/12）
- まとめると、20.37 USD のうち記事の検証の分は約 14.09 USD（69%）。残りは、LT のエージェントの Haiku（約3.18 USD、うち 0.72 は未確認）、使っていない公開 IPv4 アドレス 1.09 USD、税 1.86 USD、そのほか 0.14 USD
- 進捗表（main）の「11月の実費（記録の合計）9.08 USD」は 11/11・11/19・11/26 の3本だけの値で、12月の記事の 11/16 の検証（12/12 を含めて 5.0125 USD。12/8 の 11/21 の127件は除く）と 11/30 が入っていない
