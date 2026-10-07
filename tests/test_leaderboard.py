"""Leaderboard statistics (AGENT.md M12, 2026-09-27): the paired RMVT
bootstrap and the rank-stability rule. Pure numpy, no MuJoCo."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from mcphysbench.types import Event
from mcphysbench.stats.survival import paired_bootstrap_rmvt_delta, restricted_mean_validity, bootstrap_rmvt


def _pop(times, shift=0.0, t_max=3.0, seeds=None):
    seeds = seeds if seeds is not None else range(len(times))
    out = []
    for s, t in zip(seeds, times):
        tt = t + shift
        out.append(Event.censor(t_max, seed=s) if tt >= t_max else Event(time=tt, risk="R3", censored=False, seed=s))
    return out


def test_event_seed_is_optional_and_carried_through_censor():
    e = Event(time=1.0, risk="R3", censored=False)
    assert e.seed is None
    assert Event.censor(3.0, seed=7).seed == 7 and Event.censor(3.0).seed is None


def test_paired_delta_matches_on_seed_and_ignores_unpaired_episodes():
    rng = np.random.default_rng(0)
    base = 1.0 + rng.exponential(0.6, 40)
    a = _pop(base, 0.0, seeds=range(40))
    b = _pop(base, 0.3, seeds=range(10, 50))          # seeds 10..39 shared, 30 pairs
    r = paired_bootstrap_rmvt_delta(a, b, 3.0, n_boot=300)
    assert r["n_pairs"] == 30
    shared_a = [e for e in a if e.seed >= 10]; shared_b = [e for e in b if e.seed < 40]
    expect = restricted_mean_validity(shared_a, 3.0) - restricted_mean_validity(shared_b, 3.0)
    assert abs(r["point"] - expect) < 1e-12
    assert r["ci_hi"] < 0 and r["significant"], r      # b is shifted LATER -> a - b negative


def test_paired_ci_is_tighter_than_independent_when_episode_variance_is_shared():
    """The reason the paired design exists: a constant per-episode shift
    with large shared between-episode variance is detected paired and
    missed by the width of two independent CIs."""
    rng = np.random.default_rng(1)
    base = 1.0 + rng.exponential(1.2, 50)
    a = _pop(base, 0.0); b = _pop(base, 0.15)
    paired = paired_bootstrap_rmvt_delta(a, b, 3.0, n_boot=400)
    ia = bootstrap_rmvt(a, 3.0, n_boot=400); ib = bootstrap_rmvt(b, 3.0, n_boot=400)
    indep_width = (ia[1] - ia[0]) + (ib[1] - ib[0])
    assert (paired["ci_hi"] - paired["ci_lo"]) < indep_width / 2
    assert paired["significant"]


def test_paired_delta_is_undefined_without_shared_seeds():
    a = _pop([1.2, 1.4], seeds=[1, 2]); b = _pop([1.2, 1.4], seeds=[3, 4])
    r = paired_bootstrap_rmvt_delta(a, b, 3.0, n_boot=10)
    assert r["n_pairs"] == 0 and np.isnan(r["point"]) and not r["significant"]


def test_stability_verdict_from_render_leaderboard():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from render_leaderboard import stability_verdict, RANK_STABLE_RHO
    stable = {"0.5x": {"m1": 2.0, "m2": 1.5, "m3": 1.2}, "1x": {"m1": 2.1, "m2": 1.6, "m3": 1.1}, "2x": {"m1": 2.5, "m2": 1.9, "m3": 1.3}}
    flip = {"0.5x": {"m1": 2.0, "m2": 1.5, "m3": 1.2}, "1x": {"m1": 2.1, "m2": 1.6, "m3": 1.1}, "2x": {"m1": 1.0, "m2": 1.9, "m3": 2.3}}
    v = stability_verdict(stable); assert v["stable"] and min(v["rho"].values()) >= RANK_STABLE_RHO
    v = stability_verdict(flip); assert not v["stable"]
    # the ceiling / v2v arms are excluded via `models`; too few rankable cells -> unstable, stated
    v = stability_verdict(stable, models=["m1", "m2"]); assert not v["stable"] and v["n_models"] == 2
    v = stability_verdict(flip, models=["m1", "m2", "m3"]); assert not v["stable"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS ", name)


def test_share_of_ceiling_uses_the_generated_window():
    """The first second is the given prefix and cannot fail (2026-10-07):
    an instant failure (RMVT ~1.03 s) must read ~0, not prefix / ceiling."""
    import render_leaderboard as rl
    assert rl.PREFIX_S == 1.0
    assert abs(rl.share_of_ceiling(1.0, 2.9) - 0.0) < 1e-9          # instant failure
    assert abs(rl.share_of_ceiling(1.035, 2.90) - 0.0184) < 1e-3    # gen4.5 on the rough ramp: 1.9%, not 36%
    assert abs(rl.share_of_ceiling(2.9, 2.9) - 1.0) < 1e-9          # the ceiling itself
    assert rl.share_of_ceiling(0.5, 2.9) == 0.0                      # never negative


def test_overall_score_is_the_mean_over_complete_cells():
    import render_leaderboard as rl
    score, n, partial = rl.overall_score([0.2, 0.4, 0.6, 0.8])
    assert abs(score - 0.5) < 1e-9 and n == 4 and partial is False
    score, n, partial = rl.overall_score([0.2, None, 0.4])           # missing cells are skipped, and 2 < 4 is partial
    assert abs(score - 0.3) < 1e-9 and n == 2 and partial is True
    assert rl.overall_score([]) == (None, 0, True)
