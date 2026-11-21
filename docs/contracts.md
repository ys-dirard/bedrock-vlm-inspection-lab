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

## anomalydino.py（12/10 の記事：手元の AnomalyDINO で異常度を出す。AWS は使わない）

参照の作法: dammsi/AnomalyDINO（commit `dd35b0f`、Apache-2.0）の `src/backbones.py`・`src/utils.py` をそのまま読み込み、`src/detection.py` の `run_anomaly_detection` と `src/post_eval.py` の `mean_top1p` の計算を写す（`detection.py` は CUDA の同期を必ず呼ぶので CPU では使えない）。DINOv2（Apache-2.0）は torch.hub で、commit `7764ea0` に固定して読む。

目的の整理（mino-problem-framing、2026-11-16）
- 目的：Bedrock を呼ばずに、良品1枚の見本から画像ごとの「不良らしさ」（異常度）を出し、後段で「異常度の上位 X% だけ Nova 2 Lite に回す」並べ替えに使う。あわせて、手元の CPU での1枚あたりの時間を測る
- 成功条件：テスト200枚すべてに有限の異常度と処理時間がある。計算が公式実装の VisA の既定（agnostic：テスト画像だけ背景を除く、見本は8方向に回転、DINOv2 ViT-S/14、短辺448、1近傍のコサイン距離、上位1%の平均）と同じ
- 拒否条件：テスト画像と見本が重なる。品目の見本が1枚でない。公式実装・DINOv2 の commit が設定と違う。異常度が NaN
- 未決（利用者が決める）：解像度（448 / 672）、CPU / GPU、回す割合を品目ごとに取るか全体で取るか、しきい値をテストの正解で決めること（plan.md の手順2）の扱い

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-A1 | pre | テストの各品目に、見本（refs の ref_rank 1〜n_refs）がちょうど n_refs 枚ある。テストと見本の image_id が1つでも重なれば止める |
| CI-A2 | post | パッチの距離 = 1 − （テスト画像のパッチと、見本（回転を含む全パッチ）のうち最も近いパッチとのコサイン類似度）。特徴は float32 で L2 正規化する。背景として除いたパッチの距離は 0（公式と同じ）。背景の除去は公式の `compute_background_mask`（PCA は svd_solver="randomized" で乱数を固定していない）なので、画像ごとに直前で `np.random.seed(42)` とし、同じ画像なら順番や再開に関係なく同じ値にする（公式との違い。2026-11-16 の試運転で、同じ8枚を2回出して完全に一致） |
| CI-A3 | post | 画像の異常度 = 全パッチ（背景の 0 を含む）N 個の距離のうち、大きい順に floor(0.01 N) 個の平均。floor(0.01 N) が 0 なら最大値（公式の mean_top1p） |
| CI-A4 | post | 1枚1行で image_id・category・label・score・time_s・n_patches・n_foreground を残す。time_s は画像の読み込みから異常度までの時間（モデルの読み込みと見本の特徴づくりは含めず、品目ごとに別に残す） |
| CI-A5 | pre | 再開のとき、異常度の中身を決める設定（model_name、resolution、masking、rotation、n_refs、公式実装と DINOv2 の commit、refs の SHA-256）が前の記録と違えば止める。済んだ image_id は飛ばす |

## prefilter.py（12/10 の記事：AnomalyDINO の異常度で絞り、上位だけ Nova 2 Lite に回す2段構成。API は呼ばない）

目的：手元の AnomalyDINO（費用 0）で並べ、異常度の高い画像だけを Nova 2 Lite に回したときの、見逃し率・誤検知率・F1・回した割合・1,000枚あたりの費用を、記録（AnomalyDINO の scores.jsonl と 11/11 の条件 B n1_B）だけから計算する。

決めたこと（2026-11-16、利用者が承認）
- F-D1：解像度は 448 を主、672 を参考に並べる。どちらも CPU
- F-D2：「見逃し率 2%・5%」のしきい値はテスト画像の正解で決めてよい。ただし楽観的な値（テストに合わせた値）であることを表に書く。あわせて、正解を使わない固定の割合（上位 10・20・30・50%）の結果も出す
- F-D3：回した画像の Nova の判定は repeat=1 を主にし、費用も1回分。3回の多数決は参考の列（費用は3回分）
- F-D4：上位 X% は品目ごと（品目の中で異常度の上位 X%）を主にし、200枚全体で取る版を参考に出す

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-F1 | pre | AnomalyDINO の記録と Nova の記録で、画像の集合・正解・品目が一致しなければ止める。AnomalyDINO は各画像ちょうど1行で異常度が有限、Nova は各画像ちょうど3回（repeat 1〜3。CI-K1 と同じ）でなければ止める |
| CI-F2 | post | Nova の1枚ごとの判定：主は repeat=1 の判定（解析失敗なら None）と、その回の cost_usd。参考は CI-K2 の多数決（nova_pred）と3回の cost_usd の和 |
| CI-F3 | post | 正解で決めるしきい値（楽観的）：不良品の異常度の値を候補とし、「異常度 < t の不良品の割合 ≤ 目標の見逃し率」を満たす最大の t（CI-M7 と同じ規則）。品目ごと（主）と全体（参考）の2通り。異常度 ≥ t を NG とする |
| CI-F4 | post | 上位 X% に回す：範囲（品目ごと／全体）の画像数 n について k = ceil(X × n / 100) 枚。異常度の高い順、同じ値なら image_id の昇順で並べた先頭 k 枚を回す。しきい値で回す条件は、異常度 ≥ t（CI-F3）の画像を回す |
| CI-F5 | post | 最終の判定：回して Nova の判定が有効ならその判定。回さない画像と、回したが Nova が解析失敗の画像は「残りの扱い」に従う：OK とする、または AnomalyDINO の判定（CI-F3 のしきい値で異常度 ≥ t なら NG） |
| CI-F6 | post | 条件ごとの指標：最終の判定に CI-M2〜M4 を当てる（1枚1判定）。回した割合 = 回した枚数 / 全枚数（Nova が解析失敗でも呼んだので数える）。1,000枚あたりの費用 = 回した画像の Nova の費用（主は repeat=1、参考は3回の和）の合計 / 全枚数 × 1,000。AnomalyDINO（手元）の費用は 0 とする（電気代・PC の費用は数えない）。費用が None の行があれば None |
| CI-F7 | post | AnomalyDINO の異常度の要約：品目ごとと全体の AUROC（不良品を陽性。scikit-learn の roc_auc_score）、品目ごと・正解ごとの件数・最小・中央値・最大。品目の間で尺度がそろうかを見るため |
| CI-F9 | pre | 参考：2段目を替える（11/11 の n1_D、11/26 の n3_claude_refB など、1枚1回の記録）。各画像ちょうど1回（repeat=1）で、画像の集合・正解・品目が AnomalyDINO と一致しなければ止める。API エラーで欠けた画像があっても止める |
| CI-F10 | post | 2段目を替えた条件：回す画像の決め方・回さない画像の扱い（OK／AnomalyDINO の判定）は CI-F4・F5 と同じ（品目ごとの上位 X%）。回した画像で2段目が解析失敗なら、回さない画像と同じ扱い（11/26 の CI-K5「Claude が解析失敗なら1段目の判定」と同じ考え方）。費用は回した画像の2段目の cost_usd（解析失敗も呼んだので数える）。「2段目だけ」の行は、11/26 の「Claude だけ」と同じく解析失敗を指標から除き（D3）、有効件数を書く（費用は全件分） |
| CI-F8 | post | 手元の時間：1枚あたりの time_s の p50・p95・平均（線形補間）と合計、品目ごとの見本の特徴づくりの時間、モデルの読み込みの時間 |

## noise.py・noise_dino.py（12/22 の記事：学習データに混ぜた不良品を見つける）

目的の整理（mino-problem-framing、2026-11-21）
- 目的：良品だけで学ぶはずの学習データに不良品が 5% 混ざったとき、Nova 2 Lite（良品の見本1枚、11/11 の条件 B と同じ）と手元の AnomalyDINO（自分以外の画像を見本にした異常度）で、混入をどれだけ見つけられるか（発見率）と、良品をどれだけ疑うか（良品を疑った割合）を測る。疑った画像を除いて AnomalyDINO の見本を作り直したとき、テスト200枚の精度がどう変わるかも測る（plan.md の手順1〜5）
- 成功条件：混入ありのデータが random_state=42 で再現でき、テスト・見本・品目の広がりの画像と重ならない。判定に使う一覧に正解が書かれていない。発見率・良品を疑った割合・取り除いたあとの精度が、記録と正解のファイルだけから出る
- 拒否条件：混入ありのデータが評価の画像と重なる。判定に使う一覧に正解（label）が入る。類似度の表から出した異常度が、直接の計算（CI-A2・A3）と食い違う
- 未決（書き手が決める）：AnomalyDINO で「疑う」の決め方（品目ごとの上位 5・10・20% を並べた）、取り除いたあとの AnomalyDINO の見本の作り方（混入ありの210枚すべてを見本にし、回転しない。12/10 の「良品1枚・8方向」とは違う）
- 観測：12/10 の記録では、Nova 2 Lite（条件 B）は VisA の不良品の 97.7% を OK と判定していた

| ID | 種類 | 条件 |
| --- | --- | --- |
| CI-N1 | post | 混入ありのデータ：品目（capsules・pcb1）ごとに、train の良品から 200枚、test の不良品から 10枚を random_state=42 で選ぶ。visa_test_200・visa_wide_120・visa_refs（全 rank）の画像は使わない。品目ごとに random_state=42 で混ぜ、中立の ID（品目/n001〜n210）を付ける。判定に使う一覧（noise_train_420.csv）の label は "unknown"。正解（元の image_id・label・injected）は noise_truth_420.csv に分ける。足りない・重なるなら止める |
| CI-N2 | post | 類似度の表：品目ごとに、行の画像（混入ありの210枚と、その品目のテスト100枚）の前景のパッチ × 見本の候補（混入ありの210枚）ごとの最大のコサイン類似度。特徴は 12/10 と同じ（公式の backbones.py、DINOv2 ViT-S/14、短辺448、行の画像だけ背景を除く、乱数は画像ごとに 42）。見本は回転しない。異常度 = 1 − （選んだ見本の候補の列の最大）を前景のパッチの距離、背景のパッチは 0 として、CI-A3 の上位1%の平均。自分以外を見本にした異常度は、自分の列を除いて計算する。device が cuda なら、類似度の表の計算を torch（GPU）で行う（sim_rows_torch。numpy の sim_rows と単体テストで一致。2026-11-22 の実測で、CPU との差は類似度 1.7e-5・異常度 7.0e-7 以下、上位 X% の画像は同じ） |
| CI-N3 | post | 発見率 = 疑った混入 / 混入、良品を疑った割合 = 疑った良品 / 良品。品目ごとと全体。Nova は NG を疑ったとし、解析失敗は疑わなかったとして件数を併記する |
| CI-N4 | post | AnomalyDINO の「疑う」：品目ごとに、自分以外を見本にした異常度の上位 X%（X = 5・10・20、k = ceil(X × n / 100)、同じ値なら image_id の昇順） |
| CI-N5 | post | 取り除いたあとの見本：見本の候補（混入ありの210枚）から、方法ごとに疑った画像を除いた残り。比べる条件は、除かない（混入あり）、正解で混入だけを除く（参考の上限）、AnomalyDINO の上位 X% を除く、Nova の NG を除く |
| CI-N6 | post | テストの精度：その品目のテスト100枚の異常度から AUROC と、見逃し率 5% 以下になるしきい値（テストの正解で決める楽観的な値、CI-F3 と同じ規則）での誤検知率 |
| CI-N7 | pre | 集計のとき、類似度の表・Nova の記録・正解の画像の集合が一致しなければ止める |

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
