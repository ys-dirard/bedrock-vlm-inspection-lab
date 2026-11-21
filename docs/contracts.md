# 契約（parse・metrics・cost）

単体テスト（`tests/`）は、各テストの docstring に検証する契約 ID（CI-…）を書いています。

## 決めたこと（2026-11-05、利用者が承認）

| ID | 決めたこと |
| --- | --- |
| D1 | 業務指標のスコアは「不良らしさ」= NG なら confidence、OK なら 1 − confidence |
| D2 | 繰り返しのある実験の指標は、回ごとに計算し、平均・最小・最大を出す |
| D3 | 解析失敗は、適合率などの指標から除き、有効件数（n_valid）と解析失敗率を同じ表に書く |

## parse.py（応答の読み取り。救済しない）

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-P1 | post | 応答の本文（前後の空白を除く）全体が JSON のオブジェクトのときだけ読み取れる。前後に文章がある、配列である、壊れている、はすべて失敗 |
| CI-P2 | post | 失敗のときは pred=None と、理由の文字列（`parse_error`）を返す。例外は投げない。JSON として読めなかったときは `parsed=None`、読めたが形が違う（schema）ときは読めた JSON を `parsed` に残す |
| CI-P3 | post | 正規表現で JSON を探し出す救済はしない。` ```json ` の囲みを外すのは `allow_code_fence=True` を明示したときだけ（既定は False） |
| CI-P4 | post | 判定のキー（既定 `judgement`）の値が `ok_values` / `ng_values` のどちらかに完全一致すれば OK / NG に直す。大文字・小文字も区別する。どちらでもなければ失敗（schema） |
| CI-P5 | post | 確信度のキーがあれば、0 以上 1 以下の数（bool は除く）でなければ失敗（schema）。キーが無いときは confidence=None で成功 |
| CI-P7 | post | `allow_trailing_code_fence=True`（既定は False）のときだけ、本文（前後の空白を除く）の**末尾**にある囲みの中身を読む。条件は、本文が ```` ``` ```` で終わり、その閉じの ```` ``` ```` が行の先頭にあり、それに対応する開きの ```` ``` ````（直前の ```` ``` ````）も行の先頭にあって情報文字列が空・`json`・`JSON` のどれかであること。開きより前の文章（説明）は読まずに捨てる。囲みの後ろに文章があるとき、囲みの形が崩れているときは失敗（`json: malformed trailing code fence`）。中身には CI-P1・P4・P5 をそのまま当てる。本文が囲みで終わっていなければ、CI-P1 のとおり本文全体を JSON として読む。囲みの無い文章から `{…}` を探し出すことはしない（CI-P3 は変わらない）。2026-11-11 決定（11/26 の記事：Claude が説明のあとに JSON の囲みを付けるため） |
| CI-P6 | post | `mode: objects`（AWS サンプルの形）：最上位の `objects` が配列で、各要素がオブジェクトで、各要素の判定のキー（`qc`）の値が `ok_values` / `ng_values` に完全一致しなければ失敗（schema）。空の配列なら OK、NG の値が1つでもあれば NG、すべて OK なら OK。confidence は常に None。defect は NG の要素の `reason` を「; 」でつないだもの。サンプルのノートブックは objects[0] の qc だけを見るが、ここでは全要素を見る |

## metrics.py（評価指標）

入力は1呼び出し1行の表です。label は `normal` / `anomaly`、pred は `OK` / `NG` / None（解析失敗）です。

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-M1 | pre | label が normal / anomaly 以外、pred が OK / NG / None 以外なら ValueError |
| CI-M2 | post | NG を陽性とし、pred が None の行を除いて TP・FP・TN・FN を数える（D3）。n_valid・n_parse_fail・parse_fail_rate（= 失敗 / 全呼び出し）も返す |
| CI-M3 | post | 適合率 = TP/(TP+FP)、再現率 = TP/(TP+FN)、誤検知率 = FP/(FP+TN)、見逃し率 = FN/(TP+FN)。分母が 0 のときは None（0/0 を 0 や 1 にしない） |
| CI-M4 | post | F1 = 2PR/(P+R)。P か R が None なら None、P+R=0 なら 0 |
| CI-M5 | post | 繰り返しのある実験は、回（repeat）ごとに CI-M2〜M4 を計算し、各指標の平均・最小・最大を出す。None の回は平均に入れず、使った回数を書く（D2） |
| CI-M6 | post | 業務指標：スコア = NG なら c、OK なら 1−c（D1。浮動小数点の誤差で同じ値が別の値にならないよう、小数9桁に丸める）。スコアが t 以上を NG とみなし、見逃し率（スコア < t の不良品 / 不良品）が目標以下になる最大の t を選ぶ。そのときの良品の通過率（スコア < t の良品 / 良品）、t、実際の見逃し率を返す。スコアが同じ良品は通過させない |
| CI-M7 | post | 有効な行に confidence が無いものが1つでもあれば、業務指標は計算せず None と理由を返す。不良品か良品が 0 件のときも None |
| CI-M8 | post | 一致率 = 全回で同じ判定だった画像 / 2回以上判定した画像。どこかの回が解析失敗なら一致としない |
| CI-M9 | post | レイテンシの p50・p95 は、解析失敗も含む全呼び出しから線形補間で出す（呼び出し自体は行われたため） |
| CI-M10 | post | 1件あたりの費用は全呼び出しの平均、1,000枚あたりはその1,000倍。費用が None の行があれば None |

## cost.py（費用）

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-C1 | post | 費用 = (入力×入力単価 + 出力×出力単価 + キャッシュ読み×その単価 + キャッシュ書き×その単価) / 1,000,000（単価は100万トークンあたりの USD） |
| CI-C2 | post | 単価が null の項目でも、そのトークン数が 0 なら 0 として計算する。トークン数が 1 以上で単価が null なら費用は None（0 にしない） |
| CI-C3 | pre | トークン数が負なら ValueError。単価表に無いモデル ID と層の組み合わせなら KeyError |
| CI-C4 | post | 予算の見張り：次の呼び出しの前に「累計 + これまでで最も高かった1呼び出しの費用」が予算を超えるなら止める（超える前に止める） |
| CI-C5 | pre | 実行の前に、入力と出力の単価が両方とも null でないことを確かめる。null なら実行を始めない |
| CI-C6 | post | 再開したときは、既にある記録の費用を累計に入れてから見張る。費用が不明な記録があれば再開しない（単価を直してから記録を作り直す） |

## run.py・bedrock.py（実行）

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-R1 | pre | 再開のとき、呼び出しの中身を決める設定（モデル ID、層、プロンプトの版と中身のハッシュ、見本の枚数、縮小、temperature、maxTokens、topP、layout、ref_note、additional_fields）が前の記録と1つでも違えば、呼ぶ前に止める。読み取りの設定（parse）は後から変えてよい |
| CI-R2 | pre | 品目ごとの差し替え（prompt_vars）が足りない品目があれば、最初の呼び出しの前に止める |
| CI-R3 | post | (image_id, repeat) ごとに、成功した記録は1つだけ使う（複数あれば最後のもの）。API のエラーで終わった記録は指標に入れず、件数を n_api_error に出す |
| CI-B2 | post | `layout: refs_first` は [見本…, 検査対象, 指示]、`layout: target_first` は [検査対象, 指示, ref_note, 見本…] の順で、1通の user メッセージにする。ref_note は target_first のときだけ使える。topP は設定したときだけ送る |
| CI-B1 | post | スロットリングなどの再試行できるエラーと、タイムアウト・接続の失敗は、指数バックオフで max_attempts まで試す。それでも失敗したら例外を投げずに api_error を返す。再試行できないエラーはすぐ api_error を返す |

## embed.py・embed_run.py（条件 E：埋め込み＋k近傍法）

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-E1 | post | 応答の `embeddings` がちょうど1つで、`embeddingType` が IMAGE、長さが設定の次元、値がすべて有限の数（bool は除く）、すべて 0 ではないときだけ埋め込みとして使う。それ以外は ValueError（課金された応答なら、トークン数を残して api_error として記録する） |
| CI-E2 | post | 費用は画像1枚あたりの単価（pricing.yaml の `per_image_usd`、STANDARD_IMAGE）で数える。応答が返った呼び出しは1枚分（埋め込みの形が不正でも課金されたものとする）、応答が返らなかった呼び出しは 0。単価が null なら実行しない。HTTP ヘッダー `x-amzn-bedrock-input-token-count` の値は記録するが、費用には使わない（画像では 0 が返ることを試運転で確認） |
| CI-E3 | post | 品目ごとに、学習用の埋め込みでコサイン距離の k 近傍（k は正の奇数）の多数決をとる。不良品の割合 p > 0.5 なら NG。confidence は NG なら p、OK なら 1 − p（業務指標のスコアが p になる） |
| CI-E4 | pre | 再開のとき、埋め込みの中身を決める設定（モデル ID、embeddingPurpose、次元、縮小）が前の記録と違えば、呼ぶ前に止める。k は後から変えてよい |
| CI-E5 | pre | 学習用とテスト用の画像 ID が1つでも重なれば、呼ぶ前に止める |
| CI-D1 | post | 学習用の分割（visa_knn_train.csv）は、良品を各品目の train から、不良品を test のうち評価用（visa_test_200・visa_wide_120）に入っていないものから選ぶ（random_state=42）。VisA の 1cls.csv の train に不良品が無いため |

## cascade.py（11/26 の記事：迷った画像だけ上位モデルに回す段階判定。API は呼ばない）

目的：PoC で「1段目の安い判定のうち、どの画像を上位モデル（Claude）に回すか」によって、F1・誤検知率・費用・レイテンシがどう変わるかを、記録済みの応答だけから比べる。1段目は Nova 2 Lite の3回（n1_B）と、埋め込みの距離（n1_E の埋め込み）の2通り。2段目は Claude を全件1回（n3_claude_refB）呼んだ記録から、回した画像の分だけを使う。

決めたこと（2026-11-11、利用者が承認）
- K-D1：Nova の確信度は全600回が 0.95 か 1.00 で、R1〜R5 ではほとんど回らないことが分かった。計画の R1〜R5 はそのまま計算し、別に埋め込みの距離で回す条件（R6）を足す
- K-D2：R6 の距離は連続の値（同じ品目の良品の見本のうち最も近いものとのコサイン距離）を使う。k=3 の投票（4段階）は使わない

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-K1 | pre | Nova・Claude・埋め込みの記録で、テスト画像の集合と各画像の正解（label）・品目が一致しなければ止める。Nova は各画像ちょうど3回（repeat 1〜3）、Claude はちょうど1回でなければ止める。API エラーで欠けた画像があっても止める（欠けを黙って除かない） |
| CI-K2 | post | Nova の1枚ごとのまとめ：有効な判定（解析失敗を除く）の多数決を `nova_pred` とする。有効な判定が無い、または OK と NG が同数なら `nova_pred` は None。`split` は「3回すべてが有効で同じ判定」でないとき True（解析失敗の回があれば割れたとみなす）。`nova_conf` は有効な確信度の平均を小数6桁に丸めたもの（0.95 の平均が 0.9499… にならないように）。確信度が1つも無ければ None |
| CI-K3 | post | 回す条件：R1 = split、R2 = nova_conf < 0.8、R3 = nova_pred が NG、R4 = R1 または R3、R5_τ = nova_conf < τ（τ = 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0）。nova_conf が None の画像は R2・R5 で回す（確信度が分からない＝迷いとみなす） |
| CI-K4 | post | R6 の距離：品目ごとに、学習用の良品（role=train、label=normal）の埋め込みとのコサイン距離の最小値を、小数9桁に丸める。しきい値は、学習用の良品それぞれについて「自分以外の学習用の良品」との最小距離を求め、その百分位数（p = 0, 25, 50, 75, 90, 95, 99, 100。線形補間）を品目ごとにとったもの。距離がしきい値以上の画像を回す。テスト画像の分布でしきい値を決めない |
| CI-K5 | post | 最終の判定：回した画像で Claude の判定が有効ならその判定。回したが Claude が解析失敗なら1段目の判定。回さない画像は1段目の判定。1段目の判定は、R1〜R5 は nova_pred、R6 は「OK」（良品に近いので回さない） |
| CI-K6 | post | 条件ごとに、最終の判定へ CI-M2〜M4 を当てる（1枚1判定。解析失敗は D3 のとおり除き、有効件数を書く）。比較の行として「Nova だけ」（nova_pred）、「Claude だけ」（Claude の判定）も出す |
| CI-K7 | post | 回した割合 = 回した画像 / 全画像（Claude が解析失敗の画像も、呼んだので数える） |
| CI-K8 | post | 1枚あたりの費用 = 1段目の費用（R1〜R5・Nova だけ：Nova 3回の cost_usd の和、R6：その画像の埋め込みの cost_usd）＋回したなら Claude の cost_usd。Claude だけ：Claude の cost_usd。1,000枚あたり = 全画像の平均 × 1,000。費用が None の行があれば None。埋め込みの見本（学習用）の費用は1回きりなので含めず、別に書く |
| CI-K9 | post | 1枚あたりのレイテンシ = 1段目（Nova 3回の latency_s の和、または埋め込みの latency_s）＋回したなら Claude の latency_s（順に呼ぶとみなす）。p50・p95 は全画像から線形補間 |
| CI-K10 | post | 直った・悪くなった：回した画像のうち、1段目の判定と Claude の判定がどちらも有効なものについて、直った = 1段目が不正解で Claude が正解、悪くなった = 1段目が正解で Claude が不正解。変わらない（両方正解・両方不正解）の数も出す |
| CI-K11 | post | 確信度の帯：nova_conf で [0, 0.5)、[0.5, 0.7)、[0.7, 0.9)、[0.9, 1.0] に分け、件数と nova_pred の正解率を出す。nova_pred が None の画像は件数に入れ、正解率の分母から除く。件数 0 の帯の正解率は None |
| CI-K12 | post | 業務指標（CI-M6 の計算を使う）：画像ごとのスコアは、Nova だけ = 有効な回の D1 スコアの平均、Claude だけ = Claude の D1 スコア、段階判定 = 回して Claude が有効なら Claude の D1 スコア、それ以外は1段目のスコア（R1〜R5 は Nova のスコア、R6 は 0）。「一番よかった段階判定」は F1 が最大の条件（同じなら1,000枚あたりの費用が安いほう） |

## search/（12/20・12/21・12/24 の記事：埋め込みと手元の検索）

目的の整理（mino-problem-framing、2026-11-21）
- 目的：画像・ページ画像・文を Nova Multimodal Embeddings で埋め込み、手元（numpy）のコサイン類似度で上位 k 件を探して、「正解が上位に入るか」を API を呼び直さずに何度でも数えられるようにする。費用は1件ごとに単価から記録する
- 前提（confirmed、2026-11-21）：Nova Multimodal Embeddings は us-east-1 だけ（Geo / Global は非対応）。東京にはマルチモーダルの埋め込みは Cohere Embed v4 しかなく、このアカウントでは Marketplace の合意が NOT_AVAILABLE。S3 Vectors などのベクトルストアは作らない（数百件なら全件の内積で足りる）
- 成功条件：計画の全件に有限・非ゼロの埋め込みがあり、費用がすべて分かっている。同じ記録から同じ順位が出る（同点は索引の並び順）
- 拒否条件：応答の形が違う（種類・次元・切り詰め）、文が 8,192 文字を超える、費用が分からない（テキストでトークン数のヘッダーが無い）、再開のときに設定や中身が前の記録と違う、クエリと索引の次元が違う
- 未決（書き手が決める）：S3 Vectors を記事で使うか（使うなら results/resources_plan.md）、次元（いまは 1024。11/11 の条件 E と、AWS のブログ「Manufacturing intelligence with Amazon Nova Multimodal Embeddings」に合わせた）

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-S1 | pre | 依頼は SINGLE_EMBEDDING。purpose は仕様の9種類のうちの1つ、次元は 256・384・1024・3072。テキストは空でなく 8,192 文字以下で truncationMode=NONE（切らずに失敗させる）。画像は detailLevel（STANDARD_IMAGE / DOCUMENT_IMAGE）と形式（png / jpeg / gif / webp）が必須で、Base64 にしたあと 25 MB 以下。外れたら呼ぶ前に ValueError |
| CI-S2 | post | 応答の embeddings がちょうど1つで、embeddingType が入力の種類（IMAGE / TEXT）、truncatedCharLength が無く、長さが設定の次元、値がすべて有限の数（bool は除く）、すべて 0 ではないときだけ使う（CI-E1 を文にも広げたもの） |
| CI-S3 | post | 費用：応答が返らなければ 0。画像は1枚あたりの単価（STANDARD_IMAGE は per_image_usd、DOCUMENT_IMAGE は per_document_image_usd。形が不正でも応答が返れば1枚分）。テキストは HTTP ヘッダー `x-amzn-bedrock-input-token-count` × per_mtok_text_usd / 100万。ヘッダーが無ければ費用不明として記録し、そこで止める。使う種類の単価が null なら呼ぶ前に止める |
| CI-S4 | post | 1件1行で embeddings.jsonl に追記する。再実行では埋め込みが取れた item_id を飛ばす（API のエラーの行は呼び直す）。累計が budget_usd を超えそうなら呼ばずに止める（CI-C4 と同じ）。試運転（--limit N）は一覧の先頭 N 行を別の名前（<name>__trialN）で回す |
| CI-S5 | pre | 再開のとき、設定（model_id、purpose、dimension、image_long_edge）が前の記録と違えば止める。取得済みの item_id の中身（文・画像のバイト列の SHA-256）が変わっていても止める |
| CI-S6 | post | 上位 k 件：両方を L2 正規化した内積（コサイン類似度）の大きい順。同点は索引の並びで先のもの（安定ソート）。exclude の item_id は候補から除く。候補が k 件に満たなければあるだけ返す。ゼロ・非有限のベクトル、次元の不一致、ID の重複は ValueError |
| CI-S7 | post | 指標：hit@k = 上位 k 件に正解が1件以上あれば 1。precision@k = 上位 k 件の正解の数 / k（候補が k 件未満でも分母は k）。RR = 最初の正解の順位の逆数（無ければ 0）。でたらめに選んだときの期待値：precision = 正解の数 / 候補の数、hit@k = 1 − C(N−R, k) / C(N, k) |
| CI-S8 | post | 12/20 の評価：正解 = クエリと同じ品目で、同じ欠陥の種類（image_anno.csv の label）の、索引の不良品。索引は「不良品だけ（300）」と「良品も混ぜる（600）」の2通り。クエリごとに hit@1・hit@k・precision@k・RR・同じ品目の割合・上位 k 件の良品の数・でたらめの期待値を出し、全体と品目ごとの平均を出す |
| CI-D20 | post | 12/20 の分割：capsules（単独のラベルが bubble だけ）と pcb4（単独のラベルの不良が 44 枚で、3種類が2枚ずつ）を除く 10 品目。品目ごとに、ラベルがちょうど1つで other でない不良品から 36 枚（random_state=42）。3枚以上ある種類だけをクエリの候補にし、多い順（同数なら名前順）に巡回して6枚をクエリにする（どのクエリにも索引に同じ種類が2枚以上残る）。残り 30 枚が索引。良品は test の normal から品目ごとに 30 枚（random_state=42）。計 索引 300・クエリ 60・良品 300 |

## 前提（assumption）

- Bedrock の usage の inputTokens には、キャッシュから読んだトークンは含まれない。キャッシュを使う記事で確かめる

## mmad（11/19 の記事：MMAD の選択式の問題）

参照の作法: jam-cc/MMAD（commit `49db95b`）の `evaluation/examples/GPT4/gpt4o.py` と `evaluation/examples/helper/summary.py`。違いは記事フォルダ（november/2026-11-19_mmad-remeasure）の `results/deviations.md`。目的の整理は同じフォルダの `framing.md`。

### 決めたこと（2026-11-05、利用者が承認）

| ID | 決めたこと |
| --- | --- |
| MD-D1 | Sonnet は `jp.anthropic.claude-sonnet-4-6`（5.5 はこのアカウントで使えない） |
| MD-D2 | 画像は全モデルで縦横とも512px 以内（gpt4o.py と同じ） |
| MD-D3 | 記号が取れなかった答えは、不正解として数える値（strict）を主にし、分母から外す値（script）も併記する |

### data.py・fetch.py

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-MD1 | pre | mmad.json は固定した commit から取り、SHA-256 が一致しなければ止める |
| CI-MD2 | post | 1行1問の表。qid = 画像のキー#番号。DS-MVTec は MVTec-AD、Object Structure・Object Details は Object Analysis にまとめる（summary.py と同じ）。is_normal は画像のキーに "good" を含むか。見本は random_templates[0]。知らない課題名・データセット名があれば止める |
| CI-MD3 | post | 各課題 per_task 問を、データセットの名前の順に均等に割り振る（150 → 38・38・37・37）。異常の判別は各データセットで良品 q//2、不良品 q−q//2。層の問題が足りなければ止める（他の層で埋めない） |
| CI-MD4 | post | 層の中は qid の順に並べてから random_state=42 で選ぶ。入力の並びによらず同じ一覧になる |
| CI-MD5 | post | 画像は scale = min(512/幅, 512/高さ)、大きさ = (int(幅×scale), int(高さ×scale))。小さい画像は拡大する。JPEG（画質95）にする |
| CI-MD6 | post | 部分集合（subsample_split）は、抜き出し済みの一覧から CI-MD3・MD4 と同じ層の分け方で選ぶ。record_id は元のまま残し、record_id の順に並べる。入力の並びによらず同じ結果 |

### prompt.py

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-MP1 | post | 指示文は `prompts/mmad_gpt4o_instruction.txt`（gpt4o.py の instruction を AST で取り出したもの、SHA-256 `634f0999…`）。文 = 指示 + 見本の説明 + "The last image is the query image" + "Following is the question list: " + 改行 + 問題文 |
| CI-MP2 | post | i 問目の文には、同じ画像の1〜i 問目の問題文を並べる（前の答えは入れない）。選択肢は mmad.json の順のまま（並べ替えない） |
| CI-MP3 | post | 画像は [見本, 検査対象] の順で文の前。Claude は Messages API の本文（オンデマンドの InvokeModel とバッチで同じもの）、Nova は Converse（layout refs_first） |

### score.py・report.py

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-MS1 | post | 答え = 応答の本文で正規表現 `\b([A-E])\b` に最後に一致した記号。無ければ None。選択肢の文で合わせ直さない |
| CI-MS2 | pre | 正解が A〜E 以外なら止める |
| CI-MS3 | post | データセットごとに課題の正答率（%）。異常の判別は、良品と不良品の正答率の平均 |
| CI-MS4 | post | 各データセットの平均 = 7課題の平均。全体の行 = 各列の、データセット間の単純平均 |
| CI-MS5 | post | strict は None を不正解として数える。script は A〜E 以外を分母から外す。記号が取れなかった割合も出す |
| CI-MS6 | post | 分母 0 のセルは None。None を含む平均は None（summary.py は 0 にするが、0 で埋めない） |
| CI-MS7 | post | 95% 信頼区間は Wilson 区間 |

### run.py・invoke.py

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-MR1 | post | 試運転（--limit N）は、一覧から等間隔に N 問。出力は runs/<name>__trial<N>/ |
| CI-MR2 | pre | 再開のとき、呼び出しの中身を決める設定（モデル ID、api、mode、見本の枚数、画像の大きさ、temperature、maxTokens、指示文と一覧の SHA-256）が前の記録と違えば止める |
| CI-MR3 | post | 成功した record_id は飛ばす。API エラーの記録も残す。予算は CI-C4 で見張る |
| CI-MR4 | post | 集計では、record_id ごとに最後の成功を使い、成功が無ければ最後の失敗を使う。API エラーの問題は正答率の分母から外し、件数を run_summary に出す |
| CI-MR5 | pre/post | 集計に subset（record_id の集合）を渡すと、その問題だけで集計する。subset の record_id が記録に1つでも無ければ止める（問題数の違うモデルを、そろえずに比べない） |
| CI-MB1 | post | Messages API の応答は text ブロックだけを本文にする（thinking は数だけ extra に残す） |
| CI-MB2 | pre | バッチの入力は recordId = record_id、modelInput = CI-MP3 の本文。100件（最小件数）未満なら作らない |
| CI-MB3 | post | バッチの出力に無い問題は api_error の記録にする。一覧に無い recordId、重複した recordId は止める。費用はバッチの単価 |

### summary.py との突き合わせ（2026-11-05、利用者の許可を得て実行）

- mmad_1050.csv の1,050問に、乱数（random.Random(0)、6割は正解・4割は選択肢から無作為）で答えを作り、本物の `caculate_accuracy_mmad`（commit `49db95b`）と `accuracy_table(mode="script")` に同じ答えを入れた。4データセット×7課題と平均の、すべてのセルの差の最大は 1.4e-14（浮動小数点の誤差）。読めない答えを含まない入力での確認なので、summary.py の削除の不具合（deviations.md の 17）は対象外
