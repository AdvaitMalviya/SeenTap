"""Statistics for a 3-to-5 participant pilot: non-parametric, effect sizes shown."""
import numpy as np

from seentap import analyze


def test_friedman_and_pairwise_wilcoxon_with_bonferroni():
    rng = np.random.default_rng(0)
    # C3 (gaze+voice) faster than the two baselines, consistently per participant
    c1 = rng.normal(4.0, 0.2, 5)
    c2 = c1 + 1.0
    c3 = c1 - 1.0
    out = analyze.compare_conditions({"C1": c1, "C2": c2, "C3": c3})
    assert set(out["pairwise"]) == {("C1", "C2"), ("C1", "C3"), ("C2", "C3")}
    assert out["omnibus"]["test"] == "friedman"
    assert 0.0 <= out["omnibus"]["p"] <= 1.0
    for pair, row in out["pairwise"].items():
        assert row["p_adj"] >= row["p"], "Bonferroni only ever inflates p"
        assert "effect_size" in row


def test_it_refuses_to_pretend_a_pilot_is_a_population_study():
    out = analyze.compare_conditions({"C1": [1, 2, 3], "C2": [2, 3, 4], "C3": [3, 4, 5]})
    assert out["n"] == 3
    assert out["pilot"] is True


def test_nine_cell_table_renders_as_markdown():
    rows = [{"density": 9, "mapping": "poly", "mean_err": 91.2, "std_err": 12.0,
             "mean_dx": 60.0, "mean_dy": 70.0, "n": 5}]
    md = analyze.markdown_table(rows)
    assert "| density |" in md and "91.2" in md


def test_latency_summary_reports_the_percentiles_that_matter():
    s = analyze.latency_summary([100, 200, 300, 400, 500])
    assert s["median"] == 300 and s["p95"] >= s["median"]
