"""Per-seed MC-PhysBench events for CACHED pixel-measured episodes, on the object
channels (R1/R3, +R2 on P3, +R6/R7 on soft scenes) at the manifest's own
lambda -- exactly rescore_population.py's scoring, but keeping every
episode's (seed, time, risk) instead of only the population summary.

Why this exists (2026-09-17): the M11 confirmatory run of cosmos3super on
ramp_descent (seeds 51-100, --save-frames) generated and cached all 50
episodes, then its `--merge-existing` write was refused because the
pre-existing n=47 file predates per-seed events. The trajectories and
frames are on the Volume; scoring them here recovers the events without
re-spending three GPU-hours. R5 needs the pixels' own null and is NOT
scored here -- stated wherever these events are used (on this scene the
live population's own R5 share was 0%).

    python3 scripts/score_cached_episodes.py --scenario ramp_descent --model cosmos3super --seeds 51-100
      -> results/events_ramp_descent_cosmos3super_cached.json
"""
import sys, pathlib, json, argparse, functools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default=None, help="a-b (inclusive) or comma list; default: every cached seed")
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()
    import rescore_population as rp
    from mcphysbench.adapters.video_utils import load_trajectory
    from mcphysbench.physics import make_backend
    from mcphysbench.detect.statistics import sigma_existence, sigma_kinematic, sigma_interpenetration, kinematic_axes_for
    from mcphysbench.detect.thresholds import estimate_threshold
    from mcphysbench.detect.events import extract_event
    from mcphysbench.phi import scene_geometry as sg
    from mcphysbench.physics import softbody as sb

    spec = rp.load_spec(args.scenario)
    cache = ROOT / "results" / "trajectories" / f"{args.scenario}_{args.model}"
    files = {int(p.stem[4:]): p for p in cache.glob("seed*.npz")}
    if args.seeds:
        if "-" in args.seeds:
            a, b = args.seeds.split("-"); want = list(range(int(a), int(b) + 1))
        else:
            want = [int(x) for x in args.seeds.split(",")]
    else:
        want = sorted(files)
    missing = [s for s in want if s not in files]
    if missing:
        sys.exit(f"missing cached seeds: {missing}")
    trajs = {s: load_trajectory(files[s]) for s in want}
    n_frames = min(t.pos.shape[0] for t in trajs.values())

    roll = make_backend("mujoco", scene=spec.scene)
    refs = [roll(spec, 1000 + i) for i in range(spec.n_reference)]
    dt = refs[0].dt
    STATS = {"R1": functools.partial(sigma_existence, obj=0),
             "R3": functools.partial(sigma_kinematic, axes=kinematic_axes_for(spec.name))}
    SOFT = bool(sg.SCENES.get(spec.name, {}).get("soft_body", False))
    STATS_PHI, phi_refs = {}, None
    if SOFT:
        from mcphysbench.detect.statistics import sigma_conservation, sigma_shape
        phi_refs = sb.load_phi_refs(sb.phi_refs_path(ROOT, spec.name))
        STATS_PHI = {"R6": sigma_conservation, "R7": sigma_shape}
    if spec.target_property == "P3" and not SOFT:
        STATS["R2"] = sigma_interpenetration
    thetas = {k: estimate_threshold(refs, fn, alpha=args.alpha)[:n_frames] for k, fn in STATS.items()}
    for k, fn in STATS_PHI.items():
        thetas[k] = estimate_threshold(phi_refs, fn, alpha=args.alpha)[:n_frames]
    t_max = n_frames * dt
    events = {}
    for s in want:
        tr = trajs[s]
        sig = {k: fn(tr, refs)[:n_frames] for k, fn in STATS.items()}
        for k, fn in STATS_PHI.items():
            sig[k] = fn(tr, phi_refs)[:n_frames]
        e = extract_event(sig, thetas, dt, t_max)
        events[str(s)] = dict(seed=s, time=e.time, risk=e.risk, censored=e.censored)
    out = dict(scenario=args.scenario, model=args.model, lam=spec.lam, n=len(events), t_max=t_max,
               channels=sorted(thetas), note="scored from cached trajectories on the object channels; R5 not scored",
               events=events)
    path = ROOT / "results" / f"events_{args.scenario}_{args.model}_cached.json"
    path.write_text(json.dumps(out, indent=1))
    fired = [v for v in events.values() if not v["censored"]]
    prof = {}
    for v in fired: prof[v["risk"]] = prof.get(v["risk"], 0) + 1
    print(f"{args.scenario}/{args.model}: {len(events)} cached episodes scored at lam={spec.lam}, t_max={t_max:.2f}s; "
          f"terminated {len(fired)}: {prof}; median event time {np.median([v['time'] for v in fired]) if fired else float('nan'):.2f}s -> {path.name}")


if __name__ == "__main__":
    main()
