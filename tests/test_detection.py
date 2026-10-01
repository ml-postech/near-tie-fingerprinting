"""The pooled statistic and the query-budget reduction (CPU)."""
import math

from robust_fp.detection import PooledCounts, uses_eligibility, z_key, z_score
from robust_fp.results.robustness import qmin, worst_z_at


def test_z_score_null_and_full():
    assert z_score(25, 100, 0.25) == 0.0
    assert math.isclose(z_score(100, 100, 0.25), 75 / math.sqrt(100 * 0.25 * 0.75))
    assert math.isnan(z_score(0, 0, 0.25))


def test_detector_keys():
    assert z_key("kgw") == "all_z" and z_key("morphmark") == "all_z" and z_key("topk") == "all_z"
    assert z_key("neartie") == "elig_z" and z_key("sweet") == "elig_z" and z_key("sweet+neartie") == "elig_z"
    assert uses_eligibility("morphmark+neartie")


def test_pooled_counts_dedup_across_responses():
    c = PooledCounts(0.25)
    c.add([((1, 2), True, True), ((2, 3), False, False), ((1, 2), True, True)])
    c.add([((1, 2), True, True), ((3, 4), True, False)])
    c.add([])
    s = c.summary()
    assert s["n_scored"] == 2
    assert s["all_n"] == 3 and s["all_green"] == round(2 / 3, 4)
    assert c.T_elig == 1 and c.G_elig == 1 and s["elig_frac"] == round(1 / 3, 3)


def test_qmin_is_the_monotone_suffix_crossing():
    arms = ["a", "b"]
    curves = {}
    z_a = {100: 5, 200: 6, 300: 3, 400: 5, 500: 6}
    z_b = {100: 9, 200: 9, 300: 9, 400: 9, 500: 9}
    for n in z_a:
        curves[("cell", "a", n)] = [z_a[n]] * 5
        curves[("cell", "b", n)] = [z_b[n]] * 5
    assert worst_z_at(curves, "cell", 100, arms=arms) == 5
    assert qmin(curves, "cell", gate=4.0, arms=arms) == 400
    assert qmin(curves, "cell", gate=7.0, arms=arms) is None
