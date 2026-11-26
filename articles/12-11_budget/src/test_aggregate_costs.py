"""aggregate_costs.py の単体テスト（python -m unittest discover -s src）。"""

import unittest

import aggregate_costs as ac


class TestPer1000(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(ac.per_1000(0.1989, 600), 0.3315, places=4)

    def test_none_units(self):
        self.assertIsNone(ac.per_1000(0.05, None))
        self.assertIsNone(ac.per_1000(0.05, 0))

    def test_negative_cost(self):
        with self.assertRaises(ValueError):
            ac.per_1000(-0.01, 10)


class TestTotals(unittest.TestCase):
    def test_rows_match_cost_total_md(self):
        self.assertEqual(ac.check_totals(ac.ROWS, ac.STATED_TOTALS), [])

    def test_mismatch_is_reported(self):
        stated = dict(ac.STATED_TOTALS, **{"11-26": 1.6})
        bad = ac.check_totals(ac.ROWS, stated)
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0].startswith("11-26"))

    def test_missing_article_is_reported(self):
        stated = dict(ac.STATED_TOTALS)
        del stated["12-12"]
        self.assertTrue(any(b.startswith("12-12") for b in ac.check_totals(ac.ROWS, stated)))

    def test_flex_split_sums_to_night1(self):
        night = [r.cost_usd for r in ac.ROWS if r.run.startswith("ac08_flex__night1")]
        self.assertAlmostEqual(sum(night), 0.033109, places=5)

    def test_known_per_1000_values(self):
        # 各記事の results に書かれた1,000枚あたりと合う（12/8 の Standard day1 0.331、11/26 は 7.428）
        def p(run):
            r = next(r for r in ac.ROWS if r.run == run)
            return ac.per_1000(r.cost_usd, r.units)
        self.assertAlmostEqual(p("ac08_standard__day1"), 0.331, places=3)
        self.assertAlmostEqual(p("n3_claude_refB"), 7.4275, places=4)
        self.assertAlmostEqual(p("ac03_C0"), 0.365, places=3)

    def test_model_totals_skip_zero_and_sort(self):
        mt = ac.model_totals(ac.ROWS)
        self.assertNotIn("―", mt)
        costs = [c for c, _ in mt.values()]
        self.assertEqual(costs, sorted(costs, reverse=True))


if __name__ == "__main__":
    unittest.main()
