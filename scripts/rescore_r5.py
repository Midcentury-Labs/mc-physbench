"""Re-score a pixel-measured population with the FIXED R5 calibration
(frame_consistency.make_frame_invariance_channel, 2026-09-24; AGENT.md T11)
from its saved frames and cached trajectories -- no GPU, no model call.

Why this exists: every population scored before 2026-09-24 carries an R5
threshold calibrated over the whole clip, where the zero-by-definition
prefix frames degenerated the per-bin scale and inflated theta_R5 by up to
four orders of magnitude (collision 197 px, occlusion_reemergence 12 px,
ramp_descent 7.7 px against a ~0.03 px null). Under the fixed calibration
the null lands on the 1.0 px floor on every scene. Populations with saved
frames (--save-frames) can be re-scored here; the rest must be regenerated.

Per episode: R1/R3 (+R2 on P3, +R6/R7 on soft scenes) from the cached
trajectory exactly as rescore_population.py / score_cached_episodes.py do,
R5 from the saved frames with the object footprint projected from the
cached trajectory, then extract_event with the same precedence and
persistence. Writes results/events_<scenario>_<model>_r5fixed.json with
per-seed events under BOTH thresholds (old = the population file's own
events where the seed exists there; new = re-scored) and prints the
before/after termination profiles.

    python3 scripts/rescore_r5.py --scenario occlusion_reemergence --model cosmos3super
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
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()
    import rescore_population as rp
    from mcphysbench.adapters.video_utils import load_trajectory
    from mcphysbench.physics import make_backend
    from mcphysbench.detect.statistics import sigma_existence, sigma_kinematic, sigma_interpenetration, kinematic_axes_for
    from mcphysbench.detect.thresholds import estimate_threshold
    from mcphysbench.detect.events import extract_event
    from mcphysbench.phi import scene_geometry as sg
    from mcphysbench.phi.frame_consistency import make_frame_invariance_channel, R5_MIN_PX
    from mcphysbench.physics import softbody as sb
    from mcphysbench.render.mujoco_renderer import MujocoRenderer
    from mcphysbench.adapters.base import prefix_of
    from mcphysbench.stats.survival import validity_interval, bootstrap_vi, restricted_mean_validity, termination_profile

    spec = rp.load_spec(args.scenario)
    cfg = sg.get(args.scenario)
    fdir = ROOT / "results" / "frames" / f"{args.scenario}_{args.model}"
    tdir = ROOT / "results" / "trajectories" / f"{args.scenario}_{args.model}"
    frames_by_seed = {int(p.stem[4:]): p for p in fdir.glob("seed*.npz")}
    trajs_by_seed = {int(p.stem[4:]): p for p in tdir.glob("seed*.npz")}
    seeds = sorted(set(frames_by_seed) & set(trajs_by_seed))
    if not seeds:
        sys.exit(f"no seeds with BOTH saved frames ({fdir}) and cached trajectories ({tdir})")
    first = np.load(frames_by_seed[seeds[0]])
    n_frames, prefix_len = int(first["frames"].shape[0]), int(first["prefix_len"])

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
    print(f"R5 calibration (fixed): rendering M={spec.n_reference} references ...")
    theta_R5, sigma_R5, _ = make_frame_invariance_channel(refs, spec.scene, cfg["camera"], cfg["object_radius"],
                                                           n_frames, prefix_len, spec.fps, alpha=args.alpha)
    thetas["R5"] = theta_R5[:n_frames]
    print(f"theta_R5 (fixed) median over generated frames = {float(np.nanmedian(theta_R5[prefix_len:])):.3f} px (floor {R5_MIN_PX})")
    # Camera of the scored clips: the same renderer/camera every episode used.
    renderer = MujocoRenderer(spec.scene, height=240, width=320)
    _, gt = renderer.render(prefix_of(refs[0], 2), cameras=cfg["camera"])
    t_max = n_frames * dt

    pop_path = ROOT / "results" / f"l0_demo_{args.scenario}_{args.model}.json"
    old_events = {}
    old_theta = None
    if pop_path.exists():
        pop = json.loads(pop_path.read_text())
        old_theta = pop.get("thresholds_median", {}).get("R5")
        for e in pop.get("survival", {}).get("events", []):
            if "seed" in e:
                old_events[int(e["seed"])] = e
    events = {}
    for s in seeds:
        tr = load_trajectory(trajs_by_seed[s])
        fr = np.load(frames_by_seed[s])["frames"]
        T = min(tr.pos.shape[0], fr.shape[0], n_frames)
        sig = {k: fn(tr, refs)[:T] for k, fn in STATS.items()}
        for k, fn in STATS_PHI.items():
            sig[k] = fn(tr, phi_refs)[:T]
        sig["R5"] = sigma_R5(fr[:T], tr.pos[:T], tr.present[:T], gt.cam_pos, gt.cam_mat, gt.fovy_deg)[:T]
        th = {k: v[:T] for k, v in thetas.items()}
        e = extract_event(sig, th, dt, T * dt)
        events[s] = dict(seed=s, time=e.time, risk=e.risk, censored=e.censored,
                         r5_max_px=float(np.nanmax(sig["R5"][prefix_len:])) if T > prefix_len else float("nan"))
    from mcphysbench.types import Event
    ev = [Event(time=v["time"], risk=v["risk"], censored=v["censored"]) for v in events.values()]
    vi = validity_interval(ev); ci = bootstrap_vi(ev, n_boot=400); rm = restricted_mean_validity(ev, t_max)
    prof_new = termination_profile(ev)
    old_sub = [Event(time=v["time"], risk=v["risk"], censored=v["censored"]) for s, v in old_events.items() if s in events]
    prof_old = termination_profile(old_sub) if old_sub else {}
    changed = [s for s in seeds if s in old_events and (old_events[s]["risk"] != events[s]["risk"] or
                                                       abs(old_events[s]["time"] - events[s]["time"]) > 1e-6)]
    out = dict(scenario=args.scenario, model=args.model, lam=spec.lam, n=len(events), t_max=t_max, seeds=seeds,
               theta_R5_old_median=old_theta, theta_R5_new_median=float(np.nanmedian(theta_R5[prefix_len:])),
               vi50=vi, vi50_ci95=list(ci), rmvt=rm, termination_profile=prof_new,
               termination_profile_old_same_seeds=prof_old, n_changed=len(changed), changed_seeds=changed,
               events=events, events_old_same_seeds={str(s): old_events[s] for s in seeds if s in old_events},
               note="R5 re-scored under the 2026-09-24 calibration from saved frames; other channels from cached trajectories")
    path = ROOT / "results" / f"events_{args.scenario}_{args.model}_r5fixed.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{args.scenario}/{args.model}: n={len(events)} seeds {seeds[0]}..{seeds[-1]}; theta_R5 {old_theta} -> "
          f"{out['theta_R5_new_median']:.2f} px; VI50 {vi:.2f} {ci}; RMVT {rm:.2f}")
    print(f"  profile old (same seeds): { {k: round(v, 2) for k, v in prof_old.items()} }")
    print(f"  profile new:              { {k: round(v, 2) for k, v in prof_new.items()} }")
    print(f"  {len(changed)}/{len(events)} episodes changed event -> {path.name}")


if __name__ == "__main__":
    main()
