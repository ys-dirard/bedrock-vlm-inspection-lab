# 12/4 の記事：東京で呼べる画像対応モデルの早見表と、画像1枚のトークン数

記事（Qiita アドカレ2026 の4日目）で使ったコードと結果の表です。記事のフォルダから写したもので、検証基盤（`src/vlmlab`）とは独立しています。

- `src/token_check.py`：画像の大きさを変えて Converse API を呼び、usage の入力トークン数を記録する
- `src/summarize_token_check.py`：トークン数の表（`results/token_check_summary.csv`）を作る
- `src/image_cost_per1000.py`：画像1,000枚あたりの入力の費用（`results/image_cost_per1000.csv`）を計算する
- `src/build_cheatsheet.py`：東京で画像を入力できるモデルの早見表（`results/cheatsheet.csv`・`.md`。出典の URL つき）を作る
- `src/test_*.py`：単体テスト

値は 2026-11-16 時点です。単価・提供状況は変わることがあります。
