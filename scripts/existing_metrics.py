"""Existing fixed-horizon metrics on MC-PhysBench episodes (AGENT.md M11, 2026-09-16).

The claim to test: conventional reference-based video metrics obscure a
systematic difference that MC-PhysBench exposes. For every episode a model
generated with `--save-frames`, this scores its CONTINUATION frames
against the instrument ideal's rendered continuation of the SAME episode
(same seed, same prefix, same true physics) with the metrics reference-
based physics benchmarks use:

  * MSE / PSNR / SSIM (gaussian-window SSIM, numpy) -- frame-averaged over
    the continuation and at the LAST frame (the fixed horizon)
  * Physics-IQ-style motion-mask agreement: spatial IoU (union over time
    of "pixels that changed vs. the first continuation frame", generated
    vs. real), spatiotemporal IoU (per-frame IoU averaged), and weighted
    spatial IoU (motion-magnitude weighted)

and then compares the two models' distributions of each metric (bootstrap
CI of the difference; Mann-Whitney p) next to their MC-PhysBench fingerprints
(termination channel shares) on the same episodes. Reads
results/frames/<scenario>_<model>/seed<k>.npz (frames uint8 (T,H,W,3),
prefix_len) and results/l0_demo_<scenario>_<model>.json for the events.

    python3 scripts/existing_metrics.py --scenario occlusion_reemergence --models cosmos3nano cosmos3super
"""
import sys, pathlib, json, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _gauss1d(sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-x ** 2 / (2 * sigma ** 2)); return g / g.sum()


def _blur(img, g):
    from numpy.lib.stride_tricks import sliding_window_view as swv
    r = len(g) // 2
    p = np.pad(img, ((r, r), (r, r)), mode="reflect")
    p = (swv(p, len(g), axis=1) * g).sum(-1)
    p = (swv(p, len(g), axis=0) * g).sum(-1)
    return p


def ssim_gray(a, b, g=_gauss1d()):
    """Wang et al. SSIM on grayscale float images in [0,255]."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _blur(a, g), _blur(b, g)
    saa, sbb, sab = _blur(a * a, g) - mu_a ** 2, _blur(b * b, g) - mu_b ** 2, _blur(a * b, g) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(s.mean())


def gray(frames):
    return frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def motion_masks(frames_gray, thresh=12.0):
    """Per-frame binary masks of pixels that changed vs. the first frame
    (Physics-IQ's construction), plus the per-frame magnitude."""
    d = np.abs(frames_gray - frames_gray[:1])
    return d > thresh, d


def episode_metrics(gen, real):
    """gen, real: continuation frames uint8 (T,H,W,3), same T."""
    T = min(len(gen), len(real)); gen, real = gen[:T], real[:T]
    g, r = gray(gen), gray(real)
    mse = float(((gen.astype(np.float32) - real.astype(np.float32)) ** 2).mean())
    mse_last = float(((gen[-1].astype(np.float32) - real[-1].astype(np.float32)) ** 2).mean())
    psnr = lambda m: float(10 * np.log10(255.0 ** 2 / max(m, 1e-6)))
    ssim_t = [ssim_gray(g[t], r[t]) for t in range(0, T, 3)]
    mg, dg = motion_masks(g); mr, dr = motion_masks(r)
    ug, ur = mg.any(axis=0), mr.any(axis=0)
    spatial_iou = float((ug & ur).sum() / max((ug | ur).sum(), 1))
    st = [float((mg[t] & mr[t]).sum() / max((mg[t] | mr[t]).sum(), 1)) for t in range(T) if (mg[t] | mr[t]).any()]
    w = np.minimum(dg.max(axis=0), dr.max(axis=0)); wu = np.maximum(dg.max(axis=0), dr.max(axis=0))
    weighted_iou = float(w.sum() / max(wu.sum(), 1e-6))
    return dict(mse=mse, psnr=psnr(mse), mse_last=mse_last, psnr_last=psnr(mse_last),
                ssim=float(np.mean(ssim_t)), ssim_last=ssim_gray(g[-1], r[-1]),
                spatial_iou=spatial_iou, spatiotemporal_iou=float(np.mean(st)) if st else 0.0, weighted_iou=weighted_iou)


def load_frames(scenario, model, seed):
    p = ROOT / "results" / "frames" / f"{scenario}_{model}" / f"seed{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p); return z["frames"], int(z["prefix_len"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="occlusion_reemergence")
    ap.add_argument("--models", nargs="+", default=["cosmos3nano", "cosmos3super"])
    ap.add_argument("--reference", default="truth_render")
    ap.add_argument("--events-override", nargs="*", default=[],
                    help="model=path.json: take that model's per-seed events from a score_cached_episodes.py file "
                         "instead of its l0_demo population file")
    args = ap.parse_args()
    overrides = dict(x.split("=", 1) for x in args.events_override)
    events = {}
    for m in args.models:
        if m in overrides:
            z = json.load(open(overrides[m]))
            events[m] = {int(k): v for k, v in z["events"].items()}
            print(f"[{m}] events from {overrides[m]} ({z.get('note', '')})")
        else:
            d = json.load(open(ROOT / "results" / f"l0_demo_{args.scenario}_{m}.json"))["survival"]
            events[m] = {e["seed"]: e for e in d["events"]}
    ref_dir = ROOT / "results" / "frames" / f"{args.scenario}_{args.reference}"
    seeds = sorted(int(p.stem[4:]) for p in ref_dir.glob("seed*.npz"))
    per = {m: {} for m in args.models}
    for s in seeds:
        ref = load_frames(args.scenario, args.reference, s)
        if ref is None: continue
        rf, rp = ref
        for m in args.models:
            g = load_frames(args.scenario, m, s)
            if g is None: continue
            gf, gp = g
            per[m][s] = episode_metrics(gf[gp:], rf[rp:])
    keys = ["psnr", "ssim", "psnr_last", "ssim_last", "spatial_iou", "spatiotemporal_iou", "weighted_iou"]
    common = sorted(set.intersection(*[set(per[m]) for m in args.models]))
    print(f"scenario={args.scenario}  reference={args.reference}  matched episodes={len(common)}")
    print(f"  {'metric':20s} " + "  ".join(f"{m:>22s}" for m in args.models) + "   diff [95% CI]        MW p")
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(0)
    out = dict(scenario=args.scenario, reference=args.reference, n=len(common), metrics={}, fingerprints={})
    for k in keys:
        cols = [np.array([per[m][s][k] for s in common]) for m in args.models]
        diff = cols[1] - cols[0]
        boots = [float(np.mean(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(2000)]
        p = float(mannwhitneyu(cols[0], cols[1]).pvalue) if len(common) > 1 else float("nan")
        out["metrics"][k] = dict(means={m: float(c.mean()) for m, c in zip(args.models, cols)},
                                 sds={m: float(c.std()) for m, c in zip(args.models, cols)},
                                 diff=float(diff.mean()), diff_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], mw_p=p)
        print(f"  {k:20s} " + "  ".join(f"{c.mean():10.3f} ± {c.std():7.3f}" for c in cols)
              + f"   {diff.mean():+.3f} [{np.percentile(boots, 2.5):+.3f},{np.percentile(boots, 97.5):+.3f}]   {p:.3f}")
    # Within-model sensitivity: does the fixed-horizon metric know WHEN the
    # physics broke? Spearman correlation, per model, between each metric
    # and the MC-PhysBench event time on the same episode (censored episodes at
    # t_max). A metric that tracks physical validity should correlate
    # positively (better score <-> later failure).
    from scipy.stats import spearmanr
    print("\nWITHIN-MODEL: Spearman(metric, MC-PhysBench event time) on the same episodes")
    out["within_model_spearman"] = {}
    out["per_episode"] = {m: {str(s): dict(per[m][s], event_time=events[m][s]["time"] if s in events[m] else None,
                                            risk=events[m][s]["risk"] if s in events[m] else None) for s in common} for m in args.models}
    # Pre-registered (AGENT.md M11, 2026-09-17): the ONE confirmatory test
    # is Spearman(weighted_iou, event time) within cosmos3super, one-sided
    # H1 rho < 0, alpha 0.05. Every other within-model correlation is
    # secondary and reported with Holm correction over the family.
    PRIMARY = ("cosmos3super", "weighted_iou")
    secondary = []
    for m in args.models:
        t = np.array([events[m][s]["time"] if s in events[m] else np.nan for s in common])
        rowv = {}
        for k in keys:
            x = np.array([per[m][s][k] for s in common]); ok = np.isfinite(t) & np.isfinite(x)
            if ok.sum() > 3 and np.std(t[ok]) > 0:
                r, p = spearmanr(x[ok], t[ok])
                p_neg = p / 2 if r < 0 else 1 - p / 2          # one-sided, H1: rho < 0
                rowv[k] = dict(rho=float(r), p_two_sided=float(p), p_one_sided_neg=float(p_neg), n=int(ok.sum()))
                if (m, k) != PRIMARY:
                    secondary.append((m, k, float(p)))
        out["within_model_spearman"][m] = rowv
        print(f"  {m:14s} " + "  ".join(f"{k} rho={v['rho']:+.2f} (p={v['p_two_sided']:.2f})" for k, v in rowv.items() if k in ("psnr", "ssim", "spatial_iou", "weighted_iou")))
    prim = out["within_model_spearman"].get(PRIMARY[0], {}).get(PRIMARY[1])
    if prim:
        out["primary_test"] = dict(model=PRIMARY[0], metric=PRIMARY[1], rho=prim["rho"], n=prim["n"],
                                   p_one_sided=prim["p_one_sided_neg"], passes_alpha_0_05=bool(prim["p_one_sided_neg"] < 0.05))
        print(f"\nPRIMARY (pre-registered): Spearman({PRIMARY[1]}, event time) within {PRIMARY[0]}: rho={prim['rho']:+.3f}, "
              f"n={prim['n']}, one-sided p={prim['p_one_sided_neg']:.4f} -> {'PASS' if prim['p_one_sided_neg'] < 0.05 else 'FAIL'} at alpha=0.05")
    # Holm over the secondary family (two-sided p-values)
    secondary.sort(key=lambda x: x[2]); mth = len(secondary); holm = []
    running = 0.0
    for i, (m, k, p) in enumerate(secondary):
        adj = min(1.0, max(running, (mth - i) * p)); running = adj
        holm.append(dict(model=m, metric=k, p=p, p_holm=adj))
    out["secondary_holm"] = holm
    sig = [h for h in holm if h["p_holm"] < 0.05]
    print(f"SECONDARY (Holm over {mth} tests): {len(sig)} survive at 0.05: " + ", ".join(f"{h['model']}/{h['metric']} (p_holm={h['p_holm']:.3f})" for h in sig))
    print("\nVITALS fingerprint on the SAME episodes (terminal channel shares, VI50):")
    for m in args.models:
        ev = [events[m][s] for s in common if s in events[m]]
        n = len(ev); fired = [e for e in ev if not e["censored"]]
        prof = {}
        for e in fired: prof[e["risk"]] = prof.get(e["risk"], 0) + 1
        times = sorted(e["time"] for e in fired)
        vi50 = times[len(times) // 2] if len(fired) * 2 >= n else float("inf")
        out["fingerprints"][m] = dict(n=n, shares={k: v / n for k, v in prof.items()}, vi50=vi50)
        print(f"  {m:14s} n={n}  " + "  ".join(f"{k} {v / n * 100:.0f}%" for k, v in sorted(prof.items())) + f"   VI50 {vi50:.2f}")
    (ROOT / "results" / f"existing_metrics_{args.scenario}.json").write_text(json.dumps(out, indent=1))
    print(f"-> results/existing_metrics_{args.scenario}.json")


if __name__ == "__main__":
    main()
