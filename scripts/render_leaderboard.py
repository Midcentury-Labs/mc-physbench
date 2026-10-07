"""The MC-PhysBench leaderboard (AGENT.md M12, pre-registered 2026-09-27).

Reads results/l0_demo_<scenario>_<series>.json and
results/leaderboard_stability.json, applies the frozen rules verbatim:

  cell      n=50, seeds 1-50, t_max 3.0 s, theta_R5 <= 1.0 px (fixed
            calibration), else greyed with the defect named and unranked
  rank      per scene, by RMVT over [0, 3 s]; dense rank over PAIRED
            separations (survival.paired_bootstrap_rmvt_delta on shared
            seeds, Holm over the scene's pairs); adjacent rows whose
            paired test is not significant share a rank
  admission a scene is ranked only if Spearman rho of the RMVT ordering
            between adjacent lambda levels (0.5x->1x, 1x->2x) is >= 0.8
  cross     RMVT / ceiling RMVT grid; per model median rank (min-max),
            count of ranked scenes at-or-above constant_velocity, count
            significantly below copy_last_state -- never summed
  rows      baselines + instrument ceiling on every scene, never ranked;
            *_v2v arms in their own protocol block; superseded omitted

Writes results/leaderboard.json (every number shown) and
results/pages/leaderboard.html. No aggregate scalar anywhere (3.11).

    python3 scripts/render_leaderboard.py
"""
import sys, pathlib, json, html, math, re, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from mcphysbench.branding import PUBLIC_NAME

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAGES = RESULTS / "pages"
T_MAX = 3.0
PREFIX_S = 1.0                # the rendered prefix the model is given (30 frames at 30 fps on every scene); nothing can fail before it
                              # ends. Shares of the ceiling and the overall score use the GENERATED window, (RMVT - 1) / (ceiling - 1),
                              # so an instant failure reads ~0 (2026-10-07). Ranking still uses RMVT over [0, T_MAX]: same order.
PI = 0.3                      # events.py persistence window: a baseline event later than T_MAX - PI is censored at T_MAX
N_SEEDS = 50
RANK_STABLE_RHO = 0.8
ALPHA = 0.05
N_BOOT = 1000
N_BOOT_PAIR = 20000   # pairwise tests (2026-10-05): the p floor is 1/n_boot, and Holm multiplies it by the pair count
BASELINES = ("constant_velocity", "copy_last_state")
CEILING = "truth_render"
RIGID_SCENES = ["occlusion_reemergence", "occlusion_corridor", "ramp_descent", "ramp_descent_high_friction",
                "collision", "occlusion_corridor_interpenetration", "billiards", "block_stack"]
SOFT_SCENES = ["soft_ramp"]
HORIZON_KEYS = ("1.5", "3.0")

RULES_TEXT = """Cell: one (scenario, model) population -- n = 50, seeds 1-50 (paired across every model), t_max = 3.0 s, the manifest's own lambda, every channel scored, theta_R5 calibrated under the 2026-09-24 rule on the orchestrator's renderer, frames saved. A cell that is not all of these is shown greyed with its defect named and is not ranked.
Primary ranking statistic per scene: RMVT (restricted mean validity time over [0, 3.0 s]). VI50 is reported but does not rank (it sits at the 1.03 s detection floor for most models). Secondary, reported not ranked: S(1.5 s), S(3.0 s), termination profile.
Uncertainty and ties: 95% percentile bootstrap CI on RMVT over episodes. Every adjacent pair in a scene's ranking, and every model against both baselines and the instrument ceiling, is tested with the PAIRED bootstrap on the shared seeds; Holm correction over the pairs in that scene. Adjacent rows whose paired test is not significant share a rank (dense rank over significant separations, not a sort order).
Admission: every cell is re-scored at 0.5x / 1x / 2x the manifest lambda on the object channels; a scene is RANKED only if the Spearman rho of the RMVT ordering between adjacent levels is >= 0.8 for both steps; otherwise its rows are listed alphabetically under "not rank-stable".
Cross-scene view (no scalar): the grid of RMVT / RMVT(instrument ceiling); per model the median rank across ranked scenes with min-max, the count of ranked scenes where it is paired-indistinguishable from or better than constant_velocity, and the count where it is significantly below copy_last_state. Counts and ranks are never summed, averaged into a score, or sorted into one global order.
Rows: baselines and the instrument ceiling on every scene, never ranked, always shown. Video-conditioned arms (*_v2v) are a different protocol: own rows, never ranked against single-image rows. gemini_omni rows carry seed_reproducible = False. Superseded models are omitted.

AMENDMENT 2026-10-05 (rule v2, user decision). The admission clause above no longer withholds a rank. Every scene with at least 3 rankable cells is RANKED at the manifest lambda (itself pre-registered). The 0.5x / 1x / 2x re-score is kept and reported: a scene whose adjacent-level Spearman rho falls below 0.8 is flagged BAND-SENSITIVE, and every row on every scene shows its position at each of the three band widths, so a rank that depends on the band is visible as such rather than hidden. Reason: withholding the rank hid a real finding (a model that is approximately right for a long time but never precisely right ranks last under a strict band and first under a loose one); the per-band positions carry that finding, a withheld number did not. Every number published under the old clause was re-derived under this one on the same day."""


# The same rules for the PUBLIC page (2026-10-05, user: a public-facing
# site carries no dates, and the rules should read as a list, not a block).
# Wording is RULES_TEXT's with the dates removed; RULES_TEXT itself stays in
# leaderboard.json unchanged.
RULES_PUBLIC = [
    ("Cell", "One (scenario, model) population: n = 50, seeds 1-50 (paired across every model), t_max = 3.0 s, the manifest's own "
             "reference band, every channel scored, the frame-invariance threshold calibrated on the orchestrator's own renderer, frames saved. "
             "A cell that is not all of these is shown greyed with its defect named and is not ranked."),
    ("Ranking statistic", "Per scene, RMVT: restricted mean validity time over [0, 3.0 s]. VI50 is reported but does not rank "
             "(it sits at the 1.03 s detection floor for most models). Secondary, reported not ranked: S(1.5 s), S(3.0 s), termination profile."),
    ("Uncertainty and ties", "95% percentile bootstrap CI on RMVT over episodes. Every adjacent pair in a scene's ranking, and every model against both "
             "baselines and the instrument ceiling, is tested with the paired bootstrap on the shared seeds; Holm correction over the pairs in that scene. "
             "Adjacent rows whose paired test is not significant share a rank (a dense rank over significant separations, not a sort order)."),
    ("Original admission rule", "Every cell is re-scored at 0.5x / 1x / 2x the manifest band on the object channels; a scene was ranked only if the "
             "Spearman rho of the RMVT ordering between adjacent levels was at least 0.8 for both steps; otherwise its rows were listed alphabetically as not rank-stable."),
    ("Amendment", "The admission clause no longer withholds a rank. Every scene with at least 3 rankable cells is ranked at the manifest band (itself "
             "pre-registered). The 0.5x / 1x / 2x re-score is kept and reported: a scene whose adjacent-level rho falls below 0.8 is flagged band-specific, "
             "and every row on every scene shows its position at each of the three band widths, so a rank that depends on the band is visible as such "
             "rather than hidden. Reason: withholding the rank hid a real finding (a model that is approximately right for a long time but never "
             "precisely right ranks last under a strict band and first under a loose one); the per-band positions carry that finding, a withheld "
             "number did not. Every number published under the old clause was re-derived under this one."),
    ("Cross-scene view", "No scalar. The grid of RMVT / RMVT(instrument ceiling); per model the median rank across ranked scenes with min-max, the count "
             "of ranked scenes where it is paired-indistinguishable from or better than constant velocity, and the count where it is significantly below "
             "copy last state. Counts and ranks are never summed, averaged into a score, or sorted into one global order."),
    ("Rows", "Baselines and the instrument ceiling on every scene, never ranked, always shown. Video-conditioned arms are a different protocol: own rows, "
             "never ranked against single-image rows. Superseded models are omitted."),
    ("Display", "The first second of every episode is the given prefix and cannot fail. Ranking uses RMVT over the full window as written above, which "
             "shifts every row in a scene by the same second and changes no order; every displayed share of the ceiling and the overall score use the "
             "generated window, (RMVT - 1 s) / (ceiling RMVT - 1 s), so an instant failure reads as about zero."),
]


def share_of_ceiling(rmvt, ceiling_rmvt, prefix_s=PREFIX_S):
    """Share of the achievable validity time on the GENERATED window:
    (RMVT - prefix) / (ceiling RMVT - prefix), clipped at 0. The first
    `prefix_s` seconds are the given prefix and cannot fail, so an instant
    failure reads ~0 rather than prefix / ceiling."""
    return max(0.0, rmvt - prefix_s) / max(1e-9, ceiling_rmvt - prefix_s)


def overall_score(shares, min_scenes=4):
    """Mean share over a model's complete rigid cells; (score, n, partial).
    partial = fewer than `min_scenes` cells: shown, not placed in the overall order."""
    vals = [v for v in shares if v is not None]
    if not vals:
        return None, 0, True
    return float(np.mean(vals)), len(vals), len(vals) < min_scenes


def _events(d, seeds_from_index=False):
    from mcphysbench.types import Event
    out = []
    for i, e in enumerate(d["survival"]["events"]):
        seed = e.get("seed")
        if seed is None and seeds_from_index:
            seed = i + 1
        out.append(Event(time=float(e["time"]), risk=e.get("risk"), censored=bool(e["censored"]), seed=seed))
    return out


def _recensor(events, t_max_src):
    """Baselines are scored over the manifest horizon; re-censor at T_MAX with the persistence rule (score report's own rule)."""
    from mcphysbench.types import Event
    if abs(t_max_src - T_MAX) < 1e-9:
        return events
    out = []
    for e in events:
        if e.censored or e.time > T_MAX - PI:
            out.append(Event.censor(T_MAX, seed=e.seed))
        else:
            out.append(e)
    return out


def load_cells():
    from mcphysbench.adapters import MODEL_REGISTRY
    reg = MODEL_REGISTRY
    cells = {}
    for p in sorted(RESULTS.glob("l0_demo_*.json")):
        name = p.stem[len("l0_demo_"):]
        scen = max((s for s in RIGID_SCENES + SOFT_SCENES if name.startswith(s + "_")), key=len, default=None)
        if scen is None:
            continue
        series = name[len(scen) + 1:]
        kind = ("baseline" if series in BASELINES else "ceiling" if series == CEILING else
                "model" if series in reg else None)
        if kind is None:
            continue
        if kind == "model" and reg[series].get("superseded_by"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        ev = _events(d, seeds_from_index=(kind == "baseline"))
        defects = []
        if kind == "baseline":
            ev = _recensor(ev, float(d["t_max"]))
        else:
            if abs(float(d["t_max"]) - T_MAX) > 1e-6:
                defects.append(f"t_max {float(d['t_max']):.2f} s")
            th = (d.get("thresholds_median") or {}).get("R5")
            if th is None or th > 1.01:
                defects.append(f"R5 threshold {th:.2f} px (pre-fix)" if th else "no R5")
        seeds = [e.seed for e in ev if e.seed is not None]
        if len(ev) < N_SEEDS:
            defects.append(f"n = {len(ev)}")
        elif seeds and (min(seeds) != 1 or max(seeds) > N_SEEDS or len(set(seeds)) < N_SEEDS):
            defects.append(f"seeds {min(seeds)}-{max(seeds)}")
        ev = [e for e in ev if e.seed is None or e.seed <= N_SEEDS][:N_SEEDS] if not defects else ev
        cells[(scen, series)] = dict(scenario=scen, series=series, kind=kind, events=ev, defects=defects,
                                     protocol=reg.get(series, {}).get("conditioning", "image") if kind == "model" else None,
                                     seed_reproducible=reg.get(series, {}).get("seed_reproducible", True),
                                     r4_rate=(1 - d["long_horizon"]["censoring_rate"]) if d.get("long_horizon") and "censoring_rate" in d["long_horizon"] else None,
                                     theta_R5=(d.get("thresholds_median") or {}).get("R5"))
    return cells


def stats_for(ev):
    from mcphysbench.stats.survival import (restricted_mean_validity, bootstrap_rmvt, validity_interval, bootstrap_vi,
                                       survival_at, termination_profile)
    rm = restricted_mean_validity(ev, T_MAX)
    lo, hi = bootstrap_rmvt(ev, T_MAX, n_boot=N_BOOT)
    vi = validity_interval(ev)
    S = survival_at(ev, [1.5, 3.0])
    prof = termination_profile(ev)
    term = sum(not e.censored for e in ev) / len(ev)
    return dict(n=len(ev), rmvt=rm, rmvt_ci95=[lo, hi], vi50=vi, vi50_ci95=list(bootstrap_vi(ev, n_boot=400)),
                S_1_5=S[0], S_3_0=S[1], termination_profile=prof, terminated_frac=term,
                episodes_by_channel={k: v * term for k, v in prof.items()})


def holm(pvals):
    """Holm step-down; returns adjusted p in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def stability_verdict(rmvt_by_level, models=None):
    """rmvt_by_level: {"0.5x": {model: rmvt}, "1x": {...}, "2x": {...}} -> dict(stable, rho).
    `models`: restrict to the cells that are actually ranked (single-image,
    non-superseded, defect-free) -- the ceiling and the *_v2v arms are never
    ranked, so their positions must not decide whether a ranking is stable
    (clarified 2026-09-27 on the first dry run, AGENT.md M12)."""
    from mcphysbench.stats.survival import spearman_rank_correlation
    models = sorted(set(rmvt_by_level["0.5x"]) & set(rmvt_by_level["1x"]) & set(rmvt_by_level["2x"]) & (set(models) if models is not None else set(rmvt_by_level["1x"])))
    rho = {}
    for a, b in (("0.5x", "1x"), ("1x", "2x")):
        if len(models) < 3:
            rho[f"{a}->{b}"] = float("nan")
        else:
            rho[f"{a}->{b}"] = float(spearman_rank_correlation(np.array([rmvt_by_level[a][m] for m in models]),
                                                               np.array([rmvt_by_level[b][m] for m in models])))
    vals = list(rho.values())
    stable = bool(len(models) >= 3 and all(np.isfinite(v) and v >= RANK_STABLE_RHO for v in vals))
    return dict(stable=stable, rho=rho, n_models=len(models))


def build():
    from mcphysbench.stats.survival import paired_bootstrap_rmvt_delta
    cells = load_cells()
    stab_path = RESULTS / "leaderboard_stability.json"
    stab = json.loads(stab_path.read_text()) if stab_path.exists() else {"scenes": {}}
    board = dict(generated=datetime.datetime.now().isoformat(timespec="minutes"), rules=RULES_TEXT,
                 t_max=T_MAX, n_seeds=N_SEEDS, rank_stable_rho=RANK_STABLE_RHO, alpha=ALPHA, scenes={}, cross={})
    for scen in RIGID_SCENES + SOFT_SCENES:
        rows = {s: c for (sc, s), c in cells.items() if sc == scen}
        if not rows:
            continue
        ceiling = rows.get(CEILING)
        out_rows = {}
        for series, c in rows.items():
            st = stats_for(c["events"])
            st.update(kind=c["kind"], protocol=c["protocol"], defects=c["defects"], seed_reproducible=c["seed_reproducible"],
                      r4_rate=c["r4_rate"], theta_R5=c["theta_R5"])
            if ceiling is not None and ceiling["kind"] == "ceiling" and series != CEILING:
                c_rmvt = stats_for(ceiling["events"])["rmvt"]
                st["rmvt_over_ceiling_full"] = st["rmvt"] / c_rmvt
                st["rmvt_over_ceiling"] = share_of_ceiling(st["rmvt"], c_rmvt)
            st["after_prefix"] = max(0.0, st["rmvt"] - PREFIX_S)
            st["after_prefix_ci95"] = [max(0.0, st["rmvt_ci95"][0] - PREFIX_S), max(0.0, st["rmvt_ci95"][1] - PREFIX_S)]
            out_rows[series] = st
        # paired tests vs baselines and ceiling for every model row (any protocol)
        for series, st in out_rows.items():
            if st["kind"] != "model":
                continue
            st["vs"] = {}
            for ref in BASELINES + (CEILING,):
                if ref in rows:
                    st["vs"][ref] = paired_bootstrap_rmvt_delta(rows[series]["events"], rows[ref]["events"], T_MAX, n_boot=N_BOOT_PAIR)
        # ranking among VALID single-image model cells
        ranked_series = [s for s, st in out_rows.items() if st["kind"] == "model" and st["protocol"] == "image" and not st["defects"]]
        ranked_series.sort(key=lambda s: -out_rows[s]["rmvt"])
        pairs = [(a, b) for i, a in enumerate(ranked_series) for b in ranked_series[i + 1:]]
        tests = {f"{a}|{b}": paired_bootstrap_rmvt_delta(rows[a]["events"], rows[b]["events"], T_MAX, n_boot=N_BOOT_PAIR) for a, b in pairs}
        adj = holm([tests[k]["p_value"] for k in tests]) if tests else []
        for k, p in zip(tests, adj):
            tests[k]["p_holm"] = p
        # dense rank over significant ADJACENT separations
        rank = 1
        for i, s in enumerate(ranked_series):
            if i > 0:
                key = f"{ranked_series[i-1]}|{s}"
                if tests[key]["p_holm"] <= ALPHA:
                    rank += 1
            out_rows[s]["rank"] = rank
        sv = stab["scenes"].get(scen)
        verdict = stability_verdict(sv["rmvt"], models=ranked_series) if sv else dict(stable=False, rho={}, n_models=0, missing=True)
        verdict["ranked_cells"] = ranked_series
        # Rule v2 (2026-10-05): ranked whenever >= 3 rankable cells; the
        # stability verdict is a FLAG (band-sensitive), not a gate.
        is_ranked = len(ranked_series) >= 3
        rho_known = bool(verdict["rho"]) and all(np.isfinite(v) for v in verdict["rho"].values())
        if not rho_known:
            verdict["missing"] = True        # band check not computed for these cells: no flag, no reading
        band_sensitive = is_ranked and rho_known and not verdict["stable"]
        # Position of every rankable cell in the RMVT ordering at each band
        # width, from the same re-score the flag uses, shown per row.
        band_rank = {}
        if sv:
            for level, vals in sv["rmvt"].items():
                order = sorted((m for m in ranked_series if m in vals), key=lambda m: -vals[m])
                for i, m in enumerate(order, 1):
                    band_rank.setdefault(m, {})[level] = i
        for m, br in band_rank.items():
            out_rows[m]["band_rank"] = br
        board["scenes"][scen] = dict(rows=out_rows, ranked_order=ranked_series, pairwise=tests, stability=verdict,
                                     ranked=is_ranked, band_sensitive=band_sensitive,
                                     ceiling_rmvt=out_rows[CEILING]["rmvt"] if CEILING in out_rows else None)
    # cross-scene view
    models = sorted({s for sc in board["scenes"].values() for s, st in sc["rows"].items() if st["kind"] == "model" and st["protocol"] == "image"})
    for m in models:
        ranks, frac, ge_cv, lt_cls, n_ranked, n_cells, n_bs = [], {}, 0, 0, 0, 0, 0
        for scen, sc in board["scenes"].items():
            if scen in SOFT_SCENES:
                continue
            st = sc["rows"].get(m)
            if st is None:
                continue
            n_cells += 1
            if "rmvt_over_ceiling" in st:
                frac[scen] = st["rmvt_over_ceiling"]
            if sc["ranked"] and "rank" in st:
                n_ranked += 1
                n_bs += int(sc["band_sensitive"])
                ranks.append(st["rank"])
                v = st.get("vs", {})
                if "constant_velocity" in v and not (v["constant_velocity"]["significant"] and v["constant_velocity"]["point"] < 0):
                    ge_cv += 1
                if "copy_last_state" in v and v["copy_last_state"]["significant"] and v["copy_last_state"]["point"] < 0:
                    lt_cls += 1
        # Overall score (2026-10-06, user: an interpretability aid for the
        # website, not part of the pre-registered ranking): the mean of
        # RMVT / instrument ceiling over the model's complete rigid cells,
        # i.e. the average fraction of achievable validity time. Shown with
        # its scene count; a model with fewer than 4 complete cells is
        # listed as partial and not placed in the overall order.
        overall, overall_n, overall_partial = overall_score([frac[s] for s in frac if not board["scenes"][s]["rows"][m]["defects"]])
        board["cross"][m] = dict(n_cells=n_cells, n_ranked_scenes=n_ranked, rank_median=float(np.median(ranks)) if ranks else None,
                                 overall_score=overall, overall_n=overall_n, overall_partial=overall_partial,
                                 rank_min=min(ranks) if ranks else None, rank_max=max(ranks) if ranks else None,
                                 scenes_at_or_above_constant_velocity=ge_cv, scenes_below_copy_last_state=lt_cls,
                                 n_band_sensitive_scenes=n_bs,
                                 rmvt_over_ceiling=frac)
    # overall order: complete-coverage models by score, descending
    order = sorted((m for m, c in board["cross"].items() if c["overall_score"] is not None and not c["overall_partial"]),
                   key=lambda m: -board["cross"][m]["overall_score"])
    for i, m in enumerate(order, 1):
        board["cross"][m]["overall_rank"] = i
    return board


# ------------------------------------------------------------------ HTML
# Paint + frame: scripts/_site_theme.py (the Midcentury research-post look,
# 2026-10-01). Channel colours are the user's palette (2026-09-17), shared
# with the PDF and the population pages -- a data encoding, kept as is.
import _site_theme as T
RISK_COLOR = {"R1": "#7C3AED", "R2": "#EAB308", "R3": "#2563EB", "R4": "#0F766E", "R5": "#DC2626", "R6": "#F97316", "R7": "#EC4899"}
VALID_COLOR = "#A9FF3C"      # still valid at t_max: the accent
CHANNEL_LABEL = {"R1": "existence", "R2": "interpenetration", "R3": "kinematic", "R4": "long-horizon", "R5": "frame invariance",
                 "R6": "conservation", "R7": "shape"}
EXTRA_CSS = """
.lb th,.lb td{padding:12px 14px 12px 0}
.lb th{white-space:normal;min-width:64px;vertical-align:bottom}
.lb td.name{white-space:normal;max-width:260px;min-width:200px}
.lb td.num{font-variant-numeric:tabular-nums}
.lb .stack{width:140px}
.lb td.name .tag{display:inline-block;margin:6px 0 0}
.lb .d{display:block}
.lb .d .ci{display:block;margin-top:2px}
.cross th .tag{display:block;width:max-content;margin:6px 0 0}
.cross td.cell{font-variant-numeric:tabular-nums;color:var(--fg3)}
.cross td.cell.top{color:var(--accent)}
.cross td.cell.dim{color:var(--muted)}
.pairs{color:var(--muted);margin-top:14px}
.rulelist{margin:0;max-width:860px;border-top:1px solid var(--border)}
.rulelist dt{font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--accent);padding:22px 0 8px}
.rulelist dd{margin:0;padding:0 0 22px;color:var(--fg2);border-bottom:1px solid var(--border)}
.pairs b{color:var(--fg);font-weight:500}
"""


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "–"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{nd}f}"


def _stack(st):
    parts = []
    for k, v in sorted(st["episodes_by_channel"].items(), key=lambda kv: -kv[1]):
        if v > 0:
            parts.append(f'<i style="width:{v*100:.1f}%;background:{RISK_COLOR.get(k, "#888")}" title="{k} {CHANNEL_LABEL.get(k, "")} {v*100:.0f}% of episodes"></i>')
    valid = 1 - st["terminated_frac"]
    if valid > 0:
        parts.append(f'<i style="width:{valid*100:.1f}%;background:{VALID_COLOR}" title="still valid at 3 s {valid*100:.0f}%"></i>')
    lead = max(st["episodes_by_channel"].items(), key=lambda kv: kv[1], default=(None, 0))
    label = (f'<span class="ci">{lead[0]} {lead[1]*100:.0f}%</span>' if lead[0] and lead[1] >= 0.005 else "")
    if valid >= 0.005:
        label += f'<span class="ci"> · valid {valid*100:.0f}%</span>'
    return f'<div class="stack">{"".join(parts)}</div><div style="margin-top:6px">{label}</div>'


def _vs(st, ref):
    v = st.get("vs", {}).get(ref)
    if not v or not np.isfinite(v["point"]):
        return "–"
    cls = "sig" if v["significant"] and v["point"] > 0 else "neg" if v["significant"] else "ns"
    return f'<span class="d"><span class="{cls}">{v["point"]:+.2f}</span><span class="ci">[{v["ci_lo"]:+.2f}, {v["ci_hi"]:+.2f}]</span></span>'


def _row(series, st, ceiling_rmvt, ranked):
    cls = "grey" if st["defects"] else ("ref" if st["kind"] != "model" else "model")
    is_top = ranked and st.get("rank") == 1 and not st["defects"]
    if is_top:
        cls += " top"
    rank = f'<span class="rank">{st["rank"]}</span>' if ranked and "rank" in st and not st["defects"] else '<span class="rank ci">–</span>'
    name = T.esc(series)
    if st["kind"] == "ceiling":
        name += '<span class="tag">instrument ceiling</span>'
    elif st["kind"] == "baseline":
        name += '<span class="tag">baseline</span>'
    if st["kind"] == "model" and st["protocol"] == "video":
        name += '<span class="tag">video-conditioned</span>'
    # seed_reproducible stays in leaderboard.json; the page no longer tags it (user, 2026-10-04).
    if st["defects"]:
        name += f'<span class="tag err">unranked: {T.esc("; ".join(st["defects"]))}</span>'
    br = st.get("band_rank") or {}
    band = (" · ".join(str(br.get(k, "–")) for k in ("0.5x", "1x", "2x")) if br else "")
    frac = st.get("rmvt_over_ceiling", 1.0 if st["kind"] == "ceiling" else 0.0)
    bar = f'<div class="bar" title="{frac:.2f} of the achievable validity time after the prefix"><i style="width:{min(100, frac*100):.1f}%"></i></div>'
    ap = st.get("after_prefix", max(0.0, st["rmvt"] - PREFIX_S)); apc = st.get("after_prefix_ci95", [ap, ap])
    return (f'<tr class="{cls}"><td>{rank}</td><td class="name">{name}</td><td class="num">{band}</td>'
            f'<td class="num">{_f(ap)} <span class="ci">[{_f(apc[0])}, {_f(apc[1])}]</span>{bar}<div class="ci" style="margin-top:4px">RMVT {_f(st["rmvt"])} s</div></td>'
            f'<td class="num r">{_f(st["vi50"])}</td><td class="num r">{_f(st["S_1_5"])}</td><td class="num r">{_f(st["S_3_0"])}</td>'
            f'<td>{_stack(st)}</td><td class="num">{_vs(st, "constant_velocity")}</td><td class="num">{_vs(st, "copy_last_state")}</td><td class="num">{_vs(st, CEILING)}</td></tr>')


FLOOR_S = 1.03          # the detection floor (2026-09-12 finding): VI50 sits here for most models


def _stability_reading(scen, stab, ranked):
    """Why a scene is not rank-stable, read off the lambda re-score itself
    (results/leaderboard_stability.json), so the sentence updates with the
    data: how many rankable cells sit at the detection floor under the tight
    band (an ordering among floor values is noise), and the largest move
    between the manifest band and the loose band (threshold-marginal
    failures). Option 3 of the 2026-10-01 discussion: a finding, not a
    defect to tune away."""
    path = RESULTS / "leaderboard_stability.json"
    if stab["stable"] or stab.get("missing") or not path.exists() or len(ranked) < 3:
        return ""   # only band-sensitive scenes get a reading
    sv = json.loads(path.read_text())["scenes"].get(scen, {}).get("rmvt", {})
    if not all(k in sv for k in ("0.5x", "1x", "2x")):
        return ""
    rho = stab.get("rho", {})
    parts = []

    def swaps(a, b):
        """Pairs ordered one way at level a and the other way at level b."""
        ms = [m for m in ranked if m in sv[a] and m in sv[b]]
        out = []
        for i, x in enumerate(ms):
            for y in ms[i + 1:]:
                if (sv[a][x] - sv[a][y]) * (sv[b][x] - sv[b][y]) < 0:
                    hi, lo = (x, y) if sv[b][x] > sv[b][y] else (y, x)
                    out.append((abs(sv[b][hi] - sv[b][lo]), hi, lo))
        return sorted(out, reverse=True)

    def say(a, b, label):
        sw = swaps(a, b)
        if not sw:
            return None
        _, hi, lo = sw[0]
        return (f"between {label} {len(sw)} pair{'s' if len(sw) > 1 else ''} swap order, e.g. {T.esc(lo)} above {T.esc(hi)} at {a} "
                f"({sv[a][lo]:.2f} vs {sv[a][hi]:.2f} s) but below it at {b} ({sv[b][lo]:.2f} vs {sv[b][hi]:.2f} s)")

    if rho.get("0.5x->1x", 1.0) < RANK_STABLE_RHO:
        at_floor = [m for m in ranked if m in sv["0.5x"] and sv["0.5x"][m] <= FLOOR_S + 0.1]
        if len(at_floor) >= 2:
            parts.append(f"at 0.5× the band, {len(at_floor)} of {len(ranked)} rankable cells sit within 0.1 s of the {FLOOR_S:.2f} s detection floor, where ordering is noise")
        else:
            t = say("0.5x", "1x", "0.5× and 1×")
            if t:
                parts.append(t)
    if rho.get("1x->2x", 1.0) < RANK_STABLE_RHO:
        t = say("1x", "2x", "1× and 2×")
        if t:
            parts.append(t + ", failures that sit at the band edge")
    return ("Band-specific because: " + "; ".join(parts) + ".") if parts else ""


def _scene_badge(sc):
    stab = sc["stability"]
    if not sc["ranked"]:
        return f"<span class='tag'>unranked · {stab.get('n_models', 0)} rankable cells (3 needed)</span>"
    if stab.get("missing"):
        return "<span class='tag ok'>ranked</span><span class='tag'>band check pending</span>"
    rho = " / ".join(_f(v) for v in stab["rho"].values())
    if stab["stable"]:
        return f"<span class='tag ok'>general ranking · ρ {rho}</span>"
    return f"<span class='tag ok'>ranked</span><span class='tag err'>band-specific · ρ {rho}</span>"


def render_html(board):
    scenes_present = [s for s in RIGID_SCENES + SOFT_SCENES if s in board["scenes"]]
    rigid_present = [s for s in RIGID_SCENES if s in board["scenes"]]
    models = sorted({s for sc in board["scenes"].values() for s, st in sc["rows"].items() if st["kind"] == "model" and st["protocol"] == "image"})
    n_cells = sum(1 for sc in board["scenes"].values() for st in sc["rows"].values() if st["kind"] == "model")
    n_ranked = sum(1 for sc in board["scenes"].values() if sc["ranked"])
    n_bs = sum(1 for sc in board["scenes"].values() if sc["band_sensitive"])
    nav = [("overview", "Overview"), ("across", "Across scenes"), ("scenes", "Per scene"), ("rules", "Rules"), ("notes", "Notes")]
    body = []
    # 01 overview
    body.append(T.eyebrow(1, "Overview", "overview").replace('class="eyebrow"', 'class="kicker"'))
    body.append(f"<h1>{T.esc(PUBLIC_NAME)} leaderboard<span class='sub'>How long a video model's rollout stays physically valid, and what breaks first.</span></h1>")
    body.append("<p class='hook'>Per scene, models are ranked by restricted mean validity time (RMVT) over the 3 s scored window, paired on identical episodes, "
                "with ties where the paired test cannot separate adjacent rows. No cross-scene score exists by design; the cross-scene view is ranks and counts.</p>")
    body.append(T.stats_row([(f"{T.videos_scored(RESULTS):,}", "model videos scored", "every episode, every model version"),
                             (len(models), "models ranked", "single-image conditioning"),
                             (f"{n_ranked} / {len(scenes_present)}", "scenes ranked", f"{n_ranked - n_bs} general, {n_bs} band-specific"),
                             (n_cells, "populations", "model × scene cells"),
                             (f"{N_SEEDS}", "paired seeds per cell", f"t_max {T_MAX:g} s")]))
    body.append(f"<p class='note' style='margin-top:20px'>Every number on this page is in <a href='leaderboard.json'>leaderboard.json</a>.</p>")
    # 02 across scenes
    cross = board["cross"]
    body.append(T.eyebrow(2, "Across scenes", "across"))
    body.append("<h2>Across scenes</h2>")
    body.append("<p class='prose'>Per model: the median rank across the ranked scenes with its range, how many of those scenes are band-specific, the number of ranked scenes where the model is "
                "paired-indistinguishable from or better than constant velocity, and the number where it is significantly below copy-last-state. "
                "The pre-registered ranking is per scene; the overall score above is a reading aid built from the same cells.</p>")
    if cross:
        body.append("<h3>Overall score</h3>")
        body.append("<p class='prose'>A single number for readers who want one: each model's validity time after the prefix as a fraction of the instrument "
                    "ceiling's, averaged over its complete rigid-scene cells. The first second of every episode is the given prefix and cannot fail, so it is "
                    "excluded from both numerator and denominator: a model that breaks physics on its first generated frame scores about zero, and the "
                    "frozen-frame baseline sets the bar a model must clear to be doing better than not moving. It is a summary for "
                    "reading, not the ranking statistic: the per-scene tables, with their paired tests, are the measurement, and two models close in overall "
                    "score may still be clearly separated on individual scenes. Models with fewer than four complete cells are shown as partial.</p>")
        ov = sorted((m for m in cross if cross[m].get("overall_rank")), key=lambda m: cross[m]["overall_rank"])
        partial = sorted(m for m in cross if cross[m]["overall_score"] is not None and cross[m]["overall_partial"])
        body.append("<div class='scroll'><table class='lb cross overall'><thead><tr><th>overall</th><th>model</th><th class='r'>overall score</th><th class='r'>scenes</th>"
                    "<th>share of achievable validity time after the prefix</th></tr></thead><tbody>")
        for m in ov + partial:
            c = cross[m]; sc_ = c["overall_score"]
            rk = f"<span class='rank'>{c['overall_rank']}</span>" if c.get("overall_rank") else "<span class='rank ci'>–</span>"
            cls = "top" if c.get("overall_rank") == 1 else ("grey" if c["overall_partial"] else "")
            tag = "<span class='tag'>partial</span>" if c["overall_partial"] else ""
            body.append(f"<tr class='{cls}'><td>{rk}</td><td class='name'>{T.esc(m)}{tag}</td><td class='num r'>{sc_:.2f}</td><td class='num r'>{c['overall_n']}</td>"
                        f"<td><div class='bar' style='width:220px;margin:0'><i style='width:{min(100, sc_*100):.1f}%'></i></div></td></tr>")
        body.append("</tbody></table></div>")
        body.append("<h3>Ranks and counts</h3>")
        order = sorted(cross, key=lambda m: (cross[m]["rank_median"] if cross[m]["rank_median"] is not None else 99, -cross[m]["n_ranked_scenes"], m))
        body.append("<div class='scroll'><table class='lb cross'><thead><tr><th>model</th><th class='r'>ranked scenes</th><th class='r'>median rank</th><th class='r'>range</th><th class='r'>band-specific</th>"
                    "<th class='r'>≥ constant velocity</th><th class='r'>&lt; copy last state</th></tr></thead><tbody>")
        for m in order:
            c = cross[m]
            if c["rank_median"] is None:
                body.append(f"<tr class='grey'><td class='name'>{T.esc(m)}</td><td class='r'>0 / {c['n_cells']}</td><td class='r'>–</td><td class='r'>–</td><td class='r'>–</td><td class='r'>–</td><td class='r'>–</td></tr>")
                continue
            body.append(f"<tr><td class='name'>{T.esc(m)}</td><td class='num r'>{c['n_ranked_scenes']} / {c['n_cells']}</td><td class='num r'>{_f(c['rank_median'], 1)}</td>"
                        f"<td class='num r'>{c['rank_min']}–{c['rank_max']}</td><td class='num r'>{c['n_band_sensitive_scenes']} / {c['n_ranked_scenes']}</td><td class='num r'>{c['scenes_at_or_above_constant_velocity']} / {c['n_ranked_scenes']}</td>"
                        f"<td class='num r'>{c['scenes_below_copy_last_state']} / {c['n_ranked_scenes']}</td></tr>")
        body.append("</tbody></table></div>")
        body.append("<h3>Share of achievable validity time</h3>")
        body.append("<p class='prose'>One cell per scene and model: the model's validity time after the prefix divided by the instrument ceiling's, the true "
                    "continuation rendered through the same renderer, codec and tracker. Lime marks the top of a scene; band-specific scenes are dimmed.</p>")
        body.append("<div class='scroll'><table class='lb cross'><thead><tr><th>model</th>" +
                    "".join(f"<th class='r'>{T.esc(s)}{'' if not board['scenes'][s]['band_sensitive'] else '<span class=tag>band-specific</span>'}</th>" for s in rigid_present) +
                    "</tr></thead><tbody>")
        for m in order:
            cells = []
            for s in rigid_present:
                sc = board["scenes"][s]; st = sc["rows"].get(m)
                if st is None or "rmvt_over_ceiling" not in st:
                    cells.append("<td class='cell r'>–</td>"); continue
                cls = "top" if sc["ranked"] and st.get("rank") == 1 and not st["defects"] else ("dim" if sc["band_sensitive"] else "")
                cells.append(f"<td class='cell r {cls}'>{st['rmvt_over_ceiling']:.2f}</td>")
            body.append(f"<tr><td class='name'>{T.esc(m)}</td>{''.join(cells)}</tr>")
        body.append("</tbody></table></div>")
    # 03 per scene (tabs)
    body.append(T.eyebrow(3, "Per scene", "scenes"))
    body.append("<h2>One table per scene</h2>")
    body.append("<p class='prose'>Every scene with at least three eligible models is ranked at its own reference band, the tolerance around the true "
                "trajectory inside which a prediction still counts as physically valid. Each scene is then re-scored with that tolerance halved and "
                "doubled, and the result sorts ranked scenes into two kinds.</p>")
    body.append("<p class='prose'><b>General ranking.</b> The order of the models holds at all three tolerances. The ranking describes the models, "
                "not the setting, and can be read as a statement about the scene.</p>")
    body.append("<p class='prose'><b>Band-specific ranking.</b> The order changes when the tolerance changes. The rank shown is the one at the scene's "
                "own band; the position column beside it shows where each model falls at half and double width. A model that is roughly right for a "
                "long time but never precisely right will rank low under a strict tolerance and high under a loose one, and that pattern is itself a "
                "result worth reading. The reason each scene is band-specific is written under its table.</p>")
    body.append("<p class='prose'>Rank is a dense rank over significant adjacent separations (paired bootstrap on shared seeds, Holm-corrected within the scene), "
                "so rows that share a number are a tie. The position column is where the model falls in the RMVT ordering when the reference band is "
                "half, one, and twice the manifest width, the same re-score behind the band-specific flag, so a rank that depends on the band is visible as such. "
                "The headline number is validity after the prefix: the first second of every episode is the given prefix and cannot fail, so RMVT minus "
                "1 s is the time the model's own prediction stayed valid, with its 95% bootstrap CI; the bar under it is that time as a share of the instrument "
                "ceiling's, and the full-window RMVT is printed beneath. Paired deltas are "
                "RMVT(model) − RMVT(reference) in seconds with their 95% paired-bootstrap CI: <span class='sig'>lime</span> significantly better, "
                "<span class='neg'>red</span> significantly worse, <span class='ns'>grey</span> not separable.</p>")
    key = "".join(f'<span class="chip"><i style="background:{RISK_COLOR[k]}"></i>{k} {lab}</span>' for k, lab in CHANNEL_LABEL.items() if k != "R4")
    key += f'<span class="chip"><i style="background:{VALID_COLOR}"></i>still valid at {T_MAX:g} s</span>'
    body.append(f"<div class='chips' style='margin:0 0 28px'>{key}</div>")
    body.append("<div data-tabs><div class='tabs'>" + "".join(
        f"<button type='button' data-panel='scene-{T.esc(s)}'>{T.esc(s)}</button>" for s in scenes_present) + "</div>")
    for scen in scenes_present:
        sc = board["scenes"][scen]
        stab = sc["stability"]
        body.append(f"<section class='panel' id='scene-{T.esc(scen)}'><div class='head'><h3>{T.esc(scen)}</h3>{_scene_badge(sc)}</div>")
        body.append(f"<p class='note'>Instrument ceiling: {_f(max(0.0, sc['ceiling_rmvt'] - PREFIX_S))} s of validity after the prefix (RMVT {_f(sc['ceiling_rmvt'])} s)" +
                    ("" if not sc["band_sensitive"] else " · band-specific: ranked at the scene's own band; the order changes with the band width, see the position column") + ".</p>")
        reading = _stability_reading(scen, stab, sc["ranked_order"])
        if reading:
            body.append(f"<p class='note'>{reading}</p>")
        body.append("<div class='scroll'><table class='lb'><thead><tr><th>rank</th><th>series</th><th>position at 0.5× · 1× · 2× band</th><th>validity after prefix (s) [95% CI]</th><th class='r'>VI50 (s)</th>"
                    "<th class='r'>S(1.5)</th><th class='r'>S(3.0)</th><th>episodes by terminal channel</th><th>Δ vs constant velocity</th><th>Δ vs copy last state</th><th>Δ vs ceiling</th></tr></thead><tbody>")
        rows = sc["rows"]
        ranked = sc["ranked"]
        model_rows = list(sc["ranked_order"])
        for s in model_rows:
            body.append(_row(s, rows[s], sc["ceiling_rmvt"], ranked))
        others = [s for s in rows if s not in model_rows and rows[s]["kind"] == "model"]
        for s in sorted(others, key=lambda s: (rows[s]["protocol"] != "video", s)):
            body.append(_row(s, rows[s], sc["ceiling_rmvt"], False))
        for s in (CEILING,) + BASELINES:
            if s in rows:
                body.append(_row(s, rows[s], sc["ceiling_rmvt"], False))
        body.append("</tbody></table></div>")
        if sc["pairwise"]:
            sig = [(k, v) for k, v in sc["pairwise"].items() if v.get("p_holm", 1) <= ALPHA]
            body.append(f"<p class='pairs'><b>{len(sig)} of {len(sc['pairwise'])}</b> model pairs separable after Holm correction" +
                        (": " + ", ".join(f"{T.esc(a)} &gt; {T.esc(b)}" if v['point'] > 0 else f"{T.esc(b)} &gt; {T.esc(a)}"
                                          for (a, b), v in ((k.split('|'), v) for k, v in sig)) if sig else "") + ".</p>")
        body.append("</section>")
    body.append("</div>")
    # 04 rules
    body.append(T.eyebrow(4, "Rules", "rules"))
    body.append("<h2>Pre-registered rules</h2>")
    body.append("<p class='prose'>Written before the full matrix existed. If a rule is changed later, the old rule stays visible with the reason for "
                "the change, and every number published under the old rule is re-derived under the new one.</p>")
    body.append("<dl class='rulelist'>" + "".join(f"<dt>{T.esc(k)}</dt><dd>{T.esc(v)}</dd>" for k, v in RULES_PUBLIC) + "</dl>")
    # 05 notes
    body.append(T.eyebrow(5, "Notes", "notes"))
    body.append("<h2>Reading the numbers</h2>")
    body.append("<p class='prose'>R5 thresholds are the null of the orchestrator's own renderer and codec (1.0 px floor); the λ re-score behind the stability check runs on the "
                "object channels (R5 is pixel-side and λ-independent). VI50 is reported on every row but does not rank: it sits at the 1.03 s detection floor for most models.</p>")
    body.append("<h3>Band-specific scenes</h3>")
    body.append("<p class='prose'>Under the tight band most models collapse to the detection floor, so the instrument cannot separate them there; under "
                "the loose band some models gain a second or more because their failures are small excursions at the band edge. Both are statements "
                "about the models on that scene. The original admission clause and its amendment are both in the rules above.</p>")
    body.append("<p class='prose'>Baselines (constant velocity, copy last state) and the instrument ceiling (the true continuation, rendered) appear on every scene and are "
                "never ranked. Video-conditioned arms are a different protocol and are never ranked against single-image rows. Soft-body (soft_ramp) is a separate table with its own channels.</p>")
    body.append(f"<p><a class='btn outline' href='index.html'>All populations ↗</a></p>")
    body.append(f"<div class='foot'>{T.esc(PUBLIC_NAME)}</div>")
    return T.shell(f"{PUBLIC_NAME} leaderboard", "index.html", T.nav_items(nav), [("index.html", "Populations"), ("leaderboard.json", "JSON")],
                   "".join(body), extra_css=EXTRA_CSS, extra_js=T.JS_TABS, wide=True,
                   head_extra=f"<meta name='render-stamp' content='{T.esc(board['generated'])}'>")


def main():
    board = build()
    (RESULTS / "leaderboard.json").write_text(json.dumps(board, indent=1, default=float))
    PAGES.mkdir(parents=True, exist_ok=True)
    (PAGES / "leaderboard.html").write_text(render_html(board))
    for scen, sc in board["scenes"].items():
        st = sc["stability"]
        print(f"{scen}: {'RANKED' if sc['ranked'] else 'unranked'}{' band-sensitive' if sc['band_sensitive'] else ''} rho={ {k: round(v, 2) for k, v in st['rho'].items()} } -> " +
              ", ".join(f"{s}#{sc['rows'][s].get('rank','-')} {sc['rows'][s]['rmvt']:.2f}" for s in sc["ranked_order"]))
    print(f"-> {RESULTS / 'leaderboard.json'}, {PAGES / 'leaderboard.html'}")


if __name__ == "__main__":
    main()
