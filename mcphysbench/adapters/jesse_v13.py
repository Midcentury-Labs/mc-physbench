"""API-access world models via the solidSF Jesse v1.3 World Rollout API
(https://jesse.solidsf.com/v1/world/rollout, 2026-10).

Follows the exact adapter contract and API-provider structure established by
`mcphysbench/adapters/runway.py` and `mcphysbench/adapters/openai_sora.py`:
- `generate_fn(prefix_frames, n_frames) -> continuation_frames`, returning
  exactly `n_frames` of `(H, W, 3)` uint8 RGB.
- Matched single-image conditioning (`conditioning="image"`): the API receives
  the LAST rendered prefix frame (`prefix_frames[-1]`) encoded as a lossless
  PNG `data:image/png;base64,...` URI plus the scenario's frozen prompt from
  `scenario_prompts.SCENARIO_PROMPTS[scenario_name]`.
- Output frames are resampled through `video_utils.resample_to_target` into
  MC-PhysBench's 30 fps / 320x240 / exact-frame-count convention before
  entering the unmodified Phi (SAM2 + depth + metric 3D reconstruction) and
  calibrated R1/R2/R3 scoring pipeline.

Two model tiers are registered in `mcphysbench/adapters/__init__.py`:
- `jesse_1_3_single` (`model="jesse-1.3-single"`): single-agent inverse
  perception and deterministic forward rollout.
- `jesse_1_3_swarm_100k` (`model="jesse-1.3-swarm-100k"`): 100,000-particle
  Bayesian physical hypothesis swarm over latent dynamics, friction,
  restitution, and occluded object states.

Setup:
    export SOLIDSF_API_KEY=...
    python3 scripts/run_model_population.py --model jesse_1_3_swarm_100k --scenario collision --seeds 1,2,3
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import urllib.error
import urllib.request

import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import _read_video, resample_to_target

DEFAULT_API_URL = "https://jesse.solidsf.com/v1/world/rollout"


def frame_to_data_uri(frame: np.ndarray) -> str:
    """(H, W, 3) uint8 -> lossless `data:image/png;base64,...` URI."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _download(url: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    urllib.request.urlretrieve(url, path)
    return path


class _SolidSFHttpClient:
    """Minimal stdlib HTTP client for `POST /v1/world/rollout`."""

    def __init__(self, api_key: str, api_url: str = DEFAULT_API_URL, timeout_s: float = 600.0):
        self.api_key = api_key
        self.api_url = api_url
        self.timeout_s = timeout_s

    def rollout(self, **payload) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "mc-physbench-jesse-adapter/1.3",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"solidSF API HTTP {e.code}: {detail}") from None
        except Exception as e:
            raise RuntimeError(f"solidSF API request failed: {type(e).__name__}: {e}") from None


def _default_client_factory(timeout_s: float = 600.0):
    api_key = os.environ.get("SOLIDSF_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SOLIDSF_API_KEY not set -- see mcphysbench/adapters/jesse_v13.py's module "
            "docstring for setup (https://jesse.solidsf.com/api)."
        )
    api_url = os.environ.get("SOLIDSF_API_URL", DEFAULT_API_URL)
    return _SolidSFHttpClient(api_key=api_key, api_url=api_url, timeout_s=timeout_s)


def make_jesse_generate_fn(
    scenario_name: str,
    model: str = "jesse-1.3-swarm-100k",
    seed: int = 0,
    target_fps: int = 30,
    target_hw: tuple[int, int] = (240, 320),
    wait_timeout_s: float = 600.0,
    client_factory=None,
    log=print,
):
    """Builds `generate_fn(prefix_frames, n_frames) -> continuation_frames`
    for one solidSF Jesse v1.3 world model tier (`jesse-1.3-single` or
    `jesse-1.3-swarm-100k`). `client_factory` is the test injection point
    (same pattern as `runway.py` and `openai_sora.py`)."""
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(
            f"no frozen prompt for scenario {scenario_name!r} -- add one to "
            f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
            f"(pre-registration)."
        )
    prompt = SCENARIO_PROMPTS[scenario_name]
    make_client = client_factory or (lambda: _default_client_factory(timeout_s=wait_timeout_s))

    def generate_fn(prefix_frames: np.ndarray, n_frames: int) -> np.ndarray:
        client = make_client()
        last_frame = np.asarray(prefix_frames[-1], dtype=np.uint8)
        h, w = target_hw
        request = dict(
            model=model,
            prompt_image=frame_to_data_uri(last_frame),
            prompt_text=prompt,
            scenario=scenario_name,
            n_frames=int(n_frames),
            fps=int(target_fps),
            width=int(w),
            height=int(h),
            seed=int(seed),
        )
        resp = client.rollout(**request)
        if not isinstance(resp, dict) or resp.get("status", "succeeded") in ("failed", "error"):
            err = resp.get("error") if isinstance(resp, dict) else resp
            log(f"[solidsf:{model}] rollout FAILED: {err}")
            raise RuntimeError(f"solidSF rollout failed for model {model!r}: {err}")

        job_id = resp.get("id", "rollout_ok")
        cost = resp.get("cost_credits")
        log(f"[solidsf:{model}] rollout {job_id} SUCCEEDED, n_frames={n_frames}, cost_credits={cost}")

        if "video_url" in resp and resp["video_url"]:
            path = _download(resp["video_url"])
            try:
                raw_frames, src_fps = _read_video(path, default_fps=target_fps)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        elif "video_base64" in resp and resp["video_base64"]:
            data = base64.b64decode(resp["video_base64"])
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(data)
                path = f.name
            try:
                raw_frames, src_fps = _read_video(path, default_fps=target_fps)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        elif "frames" in resp and resp["frames"] is not None:
            raw_frames = np.asarray(resp["frames"], dtype=np.uint8)
            src_fps = float(resp.get("fps", target_fps))
        else:
            raise RuntimeError(f"solidSF rollout {job_id} returned no video_url, video_base64, or frames")

        return resample_to_target(np.asarray(raw_frames), src_fps, target_fps, n_frames, target_hw)

    return generate_fn


def make_jesse_single_generate_fn(scenario_name: str, **kwargs):
    return make_jesse_generate_fn(scenario_name, model="jesse-1.3-single", **kwargs)


def make_jesse_swarm_100k_generate_fn(scenario_name: str, **kwargs):
    return make_jesse_generate_fn(scenario_name, model="jesse-1.3-swarm-100k", **kwargs)
