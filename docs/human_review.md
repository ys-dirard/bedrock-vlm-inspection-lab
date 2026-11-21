# 人の確認を挟む構成（12/14 の記事）

Step Functions のタスクトークンで、人の確認が要る画像だけ確認を待ち、確定した判定を DynamoDB に記録する構成です。コードは `src/vlmlab/review/`（Lambda とローカルのスクリプト）と `infra/human_review/`（CDK、Python）です。

## 問題の整理（mino-problem-framing、2026-11-21）

| 項目 | 内容 | 根拠 |
| --- | --- | --- |
| 目的 | 人の確認が要る画像だけ確認を待つ仕組みを動かし、20枚で「自動で確定した件数・人に回った件数・所要時間・費用」を記録する | 記事の plan.md の「目的」、draft.md の【結果】 |
| 前提（confirmed） | Standard の状態機械だけが `.waitForTaskToken` を使える。待ちは最長1年で、`TimeoutSeconds` を過ぎると `States.Timeout` | Step Functions のドキュメント（記事の results/sources.md） |
| 前提（confirmed） | Nova 2 Lite（temperature 0）の確信度で 0.4〜0.8 を人へ回すと、200枚のうち0枚（3回とも）。Claude Sonnet 4.6 では36枚・見逃し 0.45 | 12/12 の `results/gray_zone.csv` |
| 未決（書き手が決める） | 人へ回す規則（確信度の帯／NG と OK の抜き取り／全件）。12/12 の結論「確信度で人に回しても見逃しは残る」と、12/14 の題「グレーな画像だけ人に回す」が食い違う | 記事の results/deviations.md |
| 未決（書き手が決める） | 関数 URL の認証（既定は AWS_IAM＋ローカルの署名プロキシ） | claude-code.md「関数 URL を公開する場合は認証の方法を相談する」 |
| 成功条件 | 20件すべての実行が終わる（auto_decided / human_decided / timed_out のどれか）。人へ回った件は、人の判定が DynamoDB に残る。判定の失敗は自動で OK にならない | CI-HR6・HR12、CI-IR6 |
| 拒否条件 | 公開の入口（認証なしの URL）を相談なしに作る。正解（label）を状態機械に渡す。費用が 1記事 6 USD を超える | CLAUDE.md、claude-code.md |

回し方は Lambda の環境変数（`ROUTE_POLICY_JSON`）で切り替えるので、書き手が決めたあとも構成とテストは変わりません。

## 流れ

1. ローカルの `vlmlab review start` が、画像と見本を S3 に上げ、画像ごとに `StartExecution`（入力は image_id・バケット・キー。正解は入れない）
2. `Judge`（Lambda）：S3 の見本と画像を読み、Bedrock の Converse で判定し、`route.decide` で auto / human を決める
3. `Route`（Choice）：auto なら `RecordAuto`（DynamoDB に確定の判定）。human なら `AskHuman`
4. `AskHuman`：`aws-sdk:dynamodb:putItem.waitForTaskToken` で、確認待ち（status = pending、タスクトークン）を書いて待つ
5. 確認画面（Lambda の関数 URL）で OK / NG を押すと、`SendTaskSuccess` が返り、`RecordHuman` が人の判定を書く
6. `TimeoutSeconds`（既定 3,600 秒）を過ぎたら `RecordTimeout`（status = timed_out。確認画面に「要エスカレーション」として出す）
7. `vlmlab review export` が DynamoDB の記録とローカルの正解を突き合わせて `run_log.csv` を書く

## 契約（単体テストは tests/test_review.py と infra/human_review/tests/test_stack.py）

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-HR1 | post | 不良らしさ = NG なら c、OK なら 1 − c（小数9桁に丸める。D1 と同じ）。判定か確信度が無ければ None |
| CI-HR2 | post | confidence_band：不良らしさ < lo は自動で OK、≥ hi は自動で NG、そのあいだ・解析失敗・確信度なしは人へ（12/12 の CI-CB7 と同じ） |
| CI-HR3 | post | sample：NG と解析失敗は人へ。OK は sample_unit(seed, image_id) < rate のときだけ人へ。sample_unit は SHA-256 の先頭8バイト / 2^64 で、seed と画像 ID だけで決まる |
| CI-HR4 | pre | 回し方の設定は kind ごとのキーだけを受け付け、範囲外（lo > hi、rate が [0, 1] の外、数でない値）は ValueError。CDK の synth でも同じ検証をする |
| CI-HR5 | post | all：すべて人へ |
| CI-HR6 | pre/post | 判定の Lambda の入力は image_id・bucket・key（空でない文字列）と ref_keys（文字列の配列）。Bedrock の呼び出しが失敗したら例外を投げ、状態機械は `RecordJudgeError` → Fail に進む（自動で OK にしない） |
| CI-HR7 | post | 応答は vlmlab.parse で読む（救済しない）。見本 → 検査対象 → 指示の順で送る（layout: refs_first）。本文は 4,000 文字で切り、切ったことを残す |
| CI-HR8 | pre | 確認画面の判定は "OK" / "NG" の完全一致だけ。フォームの同じキーが2つ以上なら 400 |
| CI-HR9 | post | status が pending でタスクトークンがあるときだけ、条件付きの更新で submitting にしてから SendTaskSuccess を1回返す。2回目は 409 |
| CI-HR10 | post | SendTaskSuccess が TaskTimedOut・TaskDoesNotExist・InvalidToken なら 410 を返し、status を expired にする（timed_out が先に書かれていればそれを残す）。それ以外のエラーは投げる |
| CI-HR11 | post | 確認画面は値を HTML エスケープして出し、タスクトークンは出さない |
| CI-HR12 | pre/post | run_log は1件1行。終わっていない記録、正解の無い画像、OK / NG 以外の確定の判定があれば止める。tag で始まる実行だけを書く |
| CI-HR13 | post | 見逃し率 = 自動・人で OK に確定した不良品 / 不良品、誤検知率 = NG に確定した良品 / 良品（0件なら None）。Bedrock の費用はトークン数 × 単価（cost.call_cost）。状態遷移は経路ごとの状態の数で見積もる（実測ではない） |
| CI-HR15 | post | 20枚の一覧は visa_test_200 から品目×正解ごとに5枚（random_state=42）。`data/splits/visa_review_20.csv` |
| CI-HR16 | post | 実行名は `<tag>-<番号3桁>-<画像 ID の英数字以外を - にしたもの>`（80文字以内）。実行の入力に正解を入れない。見本は品目ごとに ref_rank 1 の1枚 |
| CI-IR1 | post | 作るのは S3・DynamoDB・Lambda 2つ・関数 URL・Step Functions・IAM ロールとポリシー・ロググループだけ。カスタムリソース、VPC、EventBridge、SNS は作らない |
| CI-IR2 | post | IAM のアクションに * を使わない。S3 は GetObject だけ |
| CI-IR3 | post | Bedrock は InvokeModel だけ。jp の推論プロファイルと、その振り分け先（ap-northeast-1・ap-northeast-3）の基盤モデルに限り、基盤モデルは推論プロファイル経由の条件を付ける |
| CI-IR4 | post | 関数 URL は AWS_IAM（公開の許可を作らない）。バケットは公開をすべて止め、TLS 以外を拒否する |
| CI-IR5 | post | DynamoDB はプロビジョンド 1/1（無料枠の対象）。テーブル・バケット・ロググループは cdk destroy で消える設定。ロググループは7日で消える |
| CI-IR6 | post | 状態機械は Standard。AskHuman に TimeoutSeconds と States.Timeout の Catch がある。Route の既定は Fail |
| CI-IR7 | post | Lambda の asset は vlmlab/__init__.py・parse.py と review の __init__.py・route.py・judge_handler.py・review_handler.py だけ |

## 動かし方（デプロイは書き手の OK のあと）

```bash
cd infra/human_review
uv sync
CDK_OUTDIR=cdk.out uv run python app.py           # synth だけ（AWS は呼ばない）
uv run pytest                                      # 構成のテスト
# 回し方を変える例: npx aws-cdk@2 deploy -c route_policy='{"kind":"sample","rate":0.5,"seed":"ac14"}'
```

確認画面は `vlmlab review proxy --url <ReviewUrl>` を起動し、ブラウザで http://127.0.0.1:8765/ を開きます（関数 URL は AWS_IAM のまま）。

## 後片付け

1. `aws s3 rm s3://<BucketName> --recursive`（バケットが空でないと cdk destroy で消えない。自動削除のカスタムリソースは作っていない）
2. 待っている実行が残っていれば `aws stepfunctions stop-execution`
3. `npx aws-cdk@2 destroy VlmHumanReview1214`
4. 消えたことの確認：`aws cloudformation describe-stacks --stack-name VlmHumanReview1214`（エラーになれば消えている）、`aws logs describe-log-groups --log-group-name-prefix VlmHumanReview1214`
