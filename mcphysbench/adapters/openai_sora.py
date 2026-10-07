"""generate_fn for OpenAI Sora 2 through the OpenAI Videos API (2026-10) --
the second API-access provider after Runway (mcphysbench/adapters/runway.py,
whose shape this follows: client factory for tests, frozen prompt,
single-image conditioning, provider failure re-raised as a plain
RuntimeError).

Billing is OpenAI's own, in dollars per generated second, on the account
behind OPENAI_API_KEY (Modal secret `vitals-openai`) -- not Runway
credits, not Modal GPU time. `sora-2-pro` is the flagship tier (the
user's flagship-only rule); `sora-2` is the standard tier and is NOT
registered.

Facts this adapter is built on, and which the first paid call must
confirm (every API model's first call here has been a debugging pass):
- `client.videos.create(model, prompt, input_reference, seconds, size)`
  starts a job; `client.videos.retrieve(id)` polls `status`
  (queued / in_progress / completed / failed); `client.videos.
  download_content(id)` returns the MP4 bytes.
- `seconds` is a small enum of whole seconds ("4", "8", "12"); the
  protocol scores 2 s, so every episode is billed at the 4 s minimum
  and trimmed, exactly as veo3.1 / seedance are at theirs.
- `size` is a fixed enum ("1280x720", "720x1280", ...) and the
  reference image must match it. Our prefix frames are 320x240 (4:3);
  16:9 is forced here, so the frame is PILLARBOXED: nearest-neighbour
  upscaled x3 to 960x720 (every source pixel an exact block, nothing
  invented) and padded with black to 1280x720. The returned video is
  cropped back to the central 960 columns before resampling to
  target_hw, so the scored geometry is the same picture the model saw.
  The black bars are a documented handicap of the API's aspect rule,
  not a scene change; if a first call shows the model treating the bars
  as content, that is the first thing to revisit.
- No `seed` parameter is documented -> registered seed_reproducible =
  False, like gemini_omni / grok / happyhorse.
- Output frame rate is read from the downloaded file's own metadata
  (imageio), with 24 fps as the fallback.
"""
import io
import math
import os
import tempfile
import time

import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import _read_video, resample_to_target

SORA_SIZE = "1280x720"
SORA_SECONDS = ("4", "8", "12")
PILLARBOX_SCALE = 3          # 320x240 -> 960x720
PILLARBOX_PAD = 160          # (1280 - 960) / 2


def pillarbox(frame):
    """(240,320,3) uint8 -> (720,1280,3): x3 nearest upscale, black side bars."""
    f = np.asarray(frame)
    up = np.repeat(np.repeat(f, PILLARBOX_SCALE, axis=0), PILLARBOX_SCALE, axis=1)
    h, w = up.shape[:2]
    W = int(SORA_SIZE.split("x")[0]); H = int(SORA_SIZE.split("x")[1])
    if (h, w) != (H, w) or w > W:
        raise ValueError(f"pillarbox expects a 240x320 frame (x{PILLARBOX_SCALE} = {H}x{W - 2 * PILLARBOX_PAD}); got {f.shape}")
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:, PILLARBOX_PAD:PILLARBOX_PAD + w] = up
    return out


def unpillarbox(frames):
    """Crop the central 960 columns back out of (T,720,1280,3)."""
    return np.asarray(frames)[:, :, PILLARBOX_PAD:PILLARBOX_PAD + PILLARBOX_SCALE * 320]


def seconds_for(n_frames, target_fps):
    need = math.ceil(n_frames / target_fps)
    for s in SORA_SECONDS:
        if int(s) >= need:
            return s
    return SORA_SECONDS[-1]


def _png_bytes(frame):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame)).save(buf, format="PNG")
    return buf.getvalue()


def _default_client_factory():
    from openai import OpenAI   # lazy: only a real call needs the SDK/key
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set -- the Modal secret `vitals-openai` carries it; "
                           "see mcphysbench/adapters/openai_sora.py's module docstring.")
    return OpenAI()


def make_sora_generate_fn(scenario_name, model="sora-2-pro", size=SORA_SIZE, target_fps=30, target_hw=(240, 320),
                          wait_timeout_s=1200, poll_s=10, client_factory=None, log=print):
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time (pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]
    make_client = client_factory or _default_client_factory

    def generate_fn(prefix_frames, n_frames):
        client = make_client()
        last_frame = np.asarray(prefix_frames[-1])           # single-image conditioning
        seconds = seconds_for(n_frames, target_fps)
        ref = ("prefix.png", _png_bytes(pillarbox(last_frame)), "image/png")
        job = client.videos.create(model=model, prompt=prompt, input_reference=ref, seconds=seconds, size=size)
        t0 = time.time()
        status = getattr(job, "status", None)
        while status not in ("completed", "failed"):
            if time.time() - t0 > wait_timeout_s:
                raise RuntimeError(f"sora job {job.id} still {status!r} after {wait_timeout_s}s")
            time.sleep(poll_s)
            job = client.videos.retrieve(job.id)
            status = getattr(job, "status", None)
        if status != "completed":
            err = getattr(job, "error", None)
            log(f"[sora:{model}] job {job.id} FAILED: {err}")
            raise RuntimeError(f"sora job {job.id} failed for model {model!r}: {err}")
        log(f"[sora:{model}] job {job.id} completed, seconds={seconds}, size={size}, wall={time.time() - t0:.0f}s")

        content = client.videos.download_content(job.id)
        data = content.read() if hasattr(content, "read") else bytes(content)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(data); path = f.name
        try:
            raw_frames, src_fps = _read_video(path, default_fps=24)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        raw_frames = unpillarbox(raw_frames)
        return resample_to_target(raw_frames, src_fps, target_fps, n_frames, target_hw)

    return generate_fn
