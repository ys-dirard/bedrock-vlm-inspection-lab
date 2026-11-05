# aws_sample_*.txt の出典

- リポジトリ: https://github.com/aws-samples/sample-generative-visual-inspection （MIT-0）
- ファイル: `Nova_Pro_Visual_Inspection.ipynb` の13番目のセル（0 から数えて）
- commit: `3c76f0c6157ee88143740918ef0878f4f7e67cbf`（2026-11-05 取得）
- `aws_sample_system.txt` は `default_system_prompt`、`aws_sample_instruction.txt` は `instruction` を、`textwrap.dedent` を通した結果のまま保存した（行末の空白も残している）
- 参照画像の前に置く文（`ref_note`）は `configs/n1_D.yaml` に書いた
- 品目に固有の記述は、指示の中の出力例（スポンジの説明）だけで、差し替えていない
