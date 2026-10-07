"""One real, paid Runway API call through `mcphysbench/adapters/runway.py` --
the API-model equivalent of `remote/call_hunyuan.py download_checkpoints`:
validate the whole path (secret -> SDK auth -> task submit -> poll ->
download -> decode -> resample to exactly n_frames) with a SINGLE call
before any population is trusted to it. Deliberately its own app, not a
function on `vitals-orchestrate`: attaching the secret there requires a
redeploy, and this was first run while a multi-hour population was live
in an orchestrator container (redeploying risks killing it).

Renders the conditioning prefix LOCALLY (MuJoCo on the dev machine, the
same renderer/camera `run_model_population.py` uses) and ships the
frames to a CPU-only container that holds the secret -- the key never
needs to exist in a local shell at all.

    modal run remote/modal_app_runway_smoke.py                 # default: gen4.5, collision, 60 frames
    modal run remote/modal_app_runway_smoke.py --model gen4_turbo
"""
import pathlib
import modal

MCPB_DIR = pathlib.Path(__file__).resolve().parent.parent / "mcphysbench"

app = modal.App("vitals-sora-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy", "pillow", "pyyaml", "imageio[ffmpeg]>=2.30", "openai>=1.100")
    .add_local_dir(str(MCPB_DIR), "/root/mcpb_pkg/mcphysbench")
)


@app.function(image=image, secrets=[modal.Secret.from_name("vitals-openai")], timeout=900)
def one_call(prefix_frames, scenario: str, model: str, n_frames: int) -> dict:
    """One paid Sora call through mcphysbench/adapters/openai_sora.py (2026-10-05):
    `model` is a registry key (sora2_pro) or a bare OpenAI model id."""
    import sys, time
    sys.path.insert(0, "/root/mcpb_pkg")
    import numpy as np
    from mcphysbench.adapters import MODEL_REGISTRY
    from mcphysbench.adapters.openai_sora import make_sora_generate_fn
    logs = []
    meta = MODEL_REGISTRY.get(model, {})
    kw = {}
    if meta.get("provider") == "openai":
        model = meta["model_id"]; kw = dict(size=meta.get("size", "1280x720"))
    gen = make_sora_generate_fn(scenario, model=model, log=logs.append, **kw)
    t0 = time.time()
    out = np.asarray(gen(np.asarray(prefix_frames), n_frames))
    elapsed = time.time() - t0
    motion = float(np.abs(out[1:].astype(np.int16) - out[:-1].astype(np.int16)).mean()) if out.shape[0] > 1 else 0.0
    return dict(model=model, scenario=scenario, shape=list(out.shape), dtype=str(out.dtype),
                elapsed_s=round(elapsed, 1), nonblank=bool(out.std() > 1.0),
                mean_abs_frame_diff=round(motion, 3), log="\n".join(logs))


@app.function(image=image, secrets=[modal.Secret.from_name("vitals-openai")], timeout=300, region="us-east")
def probe() -> dict:
    """Diagnostic (2026-10-05): which video models this key can see, and the
    exact error body a sora-2-pro create returns -- no generation is made
    unless the create succeeds, in which case the job is cancelled."""
    import os
    from openai import OpenAI
    c = OpenAI()
    out = {}
    try:
        out["models_with_sora"] = sorted(m.id for m in c.models.list() if "sora" in m.id.lower())
    except Exception as e:
        out["models_error"] = repr(e)[:300]
    for model in ("sora-2-pro", "sora-2"):
        try:
            v = c.videos.create(model=model, prompt="a red ball rolls to the right on a grey floor", seconds="4", size="1280x720")
            out[model] = f"CREATED {v.id} status={getattr(v, 'status', None)}"
            try:
                c.videos.delete(v.id)
            except Exception:
                pass
        except Exception as e:
            r = getattr(e, "response", None)
            out[model] = (f"{type(e).__name__}: {str(e)[:200]} | status={getattr(r, 'status_code', None)} "
                          f"url={getattr(getattr(r, 'request', None), 'url', None)} text={getattr(r, 'text', '')[:300]!r} "
                          f"hdrs={ {k: v for k, v in dict(getattr(r, 'headers', {}) or {}).items() if k.lower() in ('x-request-id', 'openai-organization', 'openai-project', 'openai-version', 'content-type')} }")
    try:
        lst = c.videos.with_raw_response.list()
        out["GET /videos"] = f"status={lst.status_code} org={lst.headers.get('openai-organization')} project={lst.headers.get('openai-project')}"
    except Exception as e:
        r = getattr(e, "response", None)
        out["GET /videos"] = f"{type(e).__name__} status={getattr(r, 'status_code', None)} text={getattr(r, 'text', '')[:200]!r}"
    try:
        m = c.models.with_raw_response.list()
        out["GET /models"] = f"status={m.status_code} org={m.headers.get('openai-organization')} project={m.headers.get('openai-project')}"
    except Exception as e:
        out["GET /models"] = repr(e)[:200]
    try:
        import urllib.request, json as _json
        out["egress"] = urllib.request.urlopen("https://ipinfo.io/json", timeout=10).read().decode()[:200]
    except Exception as e:
        out["egress"] = repr(e)[:100]
    out["sdk"] = __import__("openai").__version__
    return out


@app.local_entrypoint()
def probe_main():
    for k, v in probe.remote().items():
        print(f"{k}: {v}")


@app.local_entrypoint()
def main(model: str = "sora2_pro", scenario: str = "collision", n_frames: int = 60):
    import sys
    sys.path.insert(0, str(MCPB_DIR.parent))
    import yaml
    from mcphysbench.types import EpisodeSpec
    from mcphysbench.physics import make_backend
    from mcphysbench.render.mujoco_renderer import MujocoRenderer
    from mcphysbench.phi import scene_geometry as sg
    from mcphysbench.adapters.base import prefix_of

    m = yaml.safe_load(open(MCPB_DIR.parent / f"configs/manifests/{scenario}.yaml"))
    spec = EpisodeSpec(name=m["name"], scene=str(MCPB_DIR.parent / m["scene"]), target_property=m["target_property"],
                       band=m["band"], lam=m["lam"], n_reference=1, horizon_s=m["horizon_s"], fps=m["fps"],
                       perturb_mode=m.get("perturb_mode", "full"))
    full = make_backend("mujoco", scene=spec.scene)(spec, seed=1)
    prefix = prefix_of(full, 30)
    frames_obj, _ = MujocoRenderer(spec.scene, height=240, width=320).render(prefix, cameras=sg.get(scenario)["camera"])
    print(f"rendered prefix {frames_obj.rgb.shape} locally; calling sora:{model} for {n_frames} frames ...")
    result = one_call.remote(frames_obj.rgb, scenario, model, n_frames)
    for k, v in result.items():
        print(f"{k}: {v}")
