# 12/11 の記事：記事ごとの費用の記録と、請求の突き合わせ

記事で使った集計と図のコードと、記事ごとの費用の表です。記事のフォルダから写したもので、検証基盤（`src/vlmlab`）とは独立しています。

- `src/aggregate_costs.py`：各記事の費用の記録（`results/cost_total.md`）を集め、`results/cost_by_article.csv`・`.md` を作る（記事のリポジトリの中で動かす前提で、このリポジトリだけでは動きません）
- `src/plot_costs.py`：記事ごとの費用の図を作る
- `src/test_aggregate_costs.py`：単体テスト

費用は呼び出しごとのトークン数×単価で記録した値（税抜き）です。
