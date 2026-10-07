"""Rank stability across lambda for the leaderboard (AGENT.md M12, M7's
own acceptance criterion): every cell with cached trajectories is
re-scored at 0.5x / 1x / 2x the manifest's lambda on the object channels
(rescore_population.score_at_lambda), and per scene the Spearman rho of
the RMVT ordering between adjacent levels is written out. R5 is pixel-side
and lambda-independent, so this is a state-band stability check; the live
R5 events are not part of it (stated on the page).

Incremental: results/leaderboard_stability.json caches each cell keyed by
(scenario, model, n_cached, lam); re-running only scores new or changed
cells. Baselines are included from their own cached state trajectories
when present under results/trajectories/<scenario>_<baseline>/.

    python3 scripts/leaderboard_stability.py                 # every scene, every cached model
    python3 scripts/leaderboard_stability.py --scenarios occlusion_reemergence
"""
import sys, pathlib, json, argparse, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "leaderboard_stability.json"
LEVELS = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}
SKIP_MODELS = ("cosmos", "cosmos14b", "cosmos720p", "wan")   # superseded


def cells():
    tdir = ROOT / "results" / "trajectories"
    scen_names = sorted(p.stem for p in (ROOT / "configs" / "manifests").glob("*.yaml"))
    for d in sorted(tdir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        scen = max((s for s in scen_names if name.startswith(s + "_")), key=len, default=None)
        if scen is None:
            continue
        model = name[len(scen) + 1:]
        if model in SKIP_MODELS:
            continue
        files = sorted(d.glob("seed*.npz"), key=lambda p: int(p.stem[4:]))
        if len(files) < 10:
            continue
        yield scen, model, files


def spearman(a, b):
    from mcphysbench.stats.survival import spearman_rank_correlation
    return float(spearman_rank_correlation(np.asarray(a, float), np.asarray(b, float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    import rescore_population as rp
    from mcphysbench.adapters.video_utils import load_trajectory
    cache = json.loads(OUT.read_text()) if OUT.exists() else {"cells": {}, "scenes": {}}
    t0 = time.time()
    for scen, model, files in cells():
        if args.scenarios and scen not in args.scenarios:
            continue
        if args.models and model not in args.models:
            continue
        nominal = rp.load_spec(scen).lam
        key = f"{scen}|{model}"
        seeds = [int(p.stem[4:]) for p in files]
        # The stamp must change when the TRAJECTORIES change, not just their
        # count: a regenerated cell (new generations, same seeds 1-50) would
        # otherwise reuse the stale re-score. Newest mtime + total bytes.
        stamp = dict(n_cached=len(files), lam=nominal, seeds_min=min(seeds), seeds_max=max(seeds),
                     newest_mtime=int(max(p.stat().st_mtime for p in files)), total_bytes=int(sum(p.stat().st_size for p in files)))
        if key in cache["cells"] and cache["cells"][key].get("stamp") == stamp:
            continue
        trajs = [load_trajectory(p) for p in files]
        n_frames = min(t.pos.shape[0] for t in trajs)
        levels = {}
        for label, mult in LEVELS.items():
            r = rp.score_at_lambda(rp.load_spec(scen, lam=nominal * mult), trajs, n_frames)
            levels[label] = dict(lam=nominal * mult, rmvt=r["rmvt"], vi50=r["vi50"], S_at=r["S_at"],
                                 termination_profile=r["termination_profile"], n=r["n"])
        cache["cells"][key] = dict(scenario=scen, model=model, stamp=stamp, levels=levels)
        print(f"{scen}/{model}: RMVT 0.5x {levels['0.5x']['rmvt']:.2f} | 1x {levels['1x']['rmvt']:.2f} | 2x {levels['2x']['rmvt']:.2f}  ({time.time()-t0:.0f}s)")
        OUT.write_text(json.dumps(cache, indent=1))
    # per-scene Spearman between adjacent levels over every cached model
    scenes = {}
    for key, c in cache["cells"].items():
        scenes.setdefault(c["scenario"], {})[c["model"]] = c["levels"]
    for scen, by_model in scenes.items():
        models = sorted(by_model)
        rho = {}
        for a, b in (("0.5x", "1x"), ("1x", "2x")):
            rho[f"{a}->{b}"] = spearman([by_model[m][a]["rmvt"] for m in models], [by_model[m][b]["rmvt"] for m in models]) if len(models) >= 3 else float("nan")
        rho["0.5x->2x"] = spearman([by_model[m]["0.5x"]["rmvt"] for m in models], [by_model[m]["2x"]["rmvt"] for m in models]) if len(models) >= 3 else float("nan")
        cache["scenes"][scen] = dict(models=models, rho=rho,
                                     rmvt={lvl: {m: by_model[m][lvl]["rmvt"] for m in models} for lvl in LEVELS})
        print(f"{scen}: {len(models)} models, rho {{ {', '.join(f'{k}: {v:.2f}' for k, v in rho.items())} }}")
    OUT.write_text(json.dumps(cache, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
