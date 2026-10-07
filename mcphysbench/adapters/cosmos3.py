"""generate_fn for NVIDIA Cosmos 3 Nano, image-to-video (2026-09) -- plugs
into `VideoWorldModel`'s own `generate_fn` slot. The successor to
Cosmos-Predict2 (`cosmos.py`), integrated as its OWN backend name
(`cosmos3nano`) rather than a flag on `cosmos`, so its results never
overwrite the Predict2 populations under the same file name -- the same
separate-name rule `cosmos14b`/`cosmos720p` already follow.

Facts this adapter is built on (confirmed against the diffusers Cosmos 3
docs, see `remote/modal_app_cosmos3.py`'s own docstring for sources):
- **Single-image conditioning** (`image=`, frame 0 anchored to it) --
  only the prefix's LAST frame reaches the model, same structural
  limitation as Wan/Hunyuan. Cosmos 3 also has a video-to-video mode
  that conditions on leading frames; deliberately NOT used here so
  Cosmos 3 is compared within the SAME single-image setting as every
  other model (input-normalization protocol: matched settings only).
  A multi-frame-conditioned variant would be a separate backend name.
- Native fps 24, confirmed (not a guess). `num_frames` 5..400; this
  adapter requests just enough native frames to cover n_frames at
  target_fps (+1), capped at the docs' default 189 -- asking for 7.9s
  of video to keep 2s wastes real H100 time.
- The SAME frozen `scenario_prompts.SCENARIO_PROMPTS` text, wrapped in
  the model-native JSON container inside the Modal app -- no LLM prompt
  upsampling, see the app's docstring for why.
"""
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target

COSMOS3_NATIVE_FPS = 24        # confirmed (pipeline `fps=24.0`, docs export at 24)
COSMOS3_MAX_FRAMES = 189       # docs' default; the hard ceiling is 400
# Video-conditioned arm (2026-09-25): the prefix (30 frames @ 30 fps = 1.0 s)
# is resampled to the model's 24 fps and its LAST 21 native frames (0.875 s,
# = 6 latents under 4x temporal compression, 4k+1 so the clean latents cover
# whole pixel frames) are held clean. The model then sees the motion of the
# prefix, which the single-image path structurally cannot (review decision 5).
COSMOS3_V2V_CONDITION_FRAMES = 21


def requested_native_frames(n_frames, target_fps=30):
    """Native (24 fps) frames needed to cover `n_frames` at `target_fps`, +1, floored at 5, capped."""
    return min(COSMOS3_MAX_FRAMES, max(5, int(np.ceil(n_frames * COSMOS3_NATIVE_FPS / target_fps)) + 1))


def make_cosmos3_generate_fn_modal(scenario_name, app_name="vitals-cosmos3", function_name="image2video",
                                   target_fps=30, target_hw=(240, 320), seed=0,
                                   num_inference_steps=35, guidance_scale=6.0, conditioning="image"):
    """Builds `generate_fn(prefix_frames, n_frames) -> continuation_frames`
    calling the deployed `vitals-cosmos3` app. Needs `modal deploy remote/
    modal_app_cosmos3.py` + `download_checkpoints` first.

    `conditioning="image"`: the prefix's last frame only (the protocol every
    other model runs under). `conditioning="video"`: the prefix's last
    COSMOS3_V2V_CONDITION_FRAMES native frames, via the app's `video2video`
    function; the returned clip's conditioning span is discarded and only
    the generated remainder is resampled to the target, so the continuation
    contract is identical to the image path's."""
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
                        f"(pre-registration).")
    if conditioning not in ("image", "video"):
        raise ValueError(f"conditioning must be 'image' or 'video', got {conditioning!r}")
    if conditioning == "video" and function_name == "image2video":
        function_name = "video2video"
    prompt = SCENARIO_PROMPTS[scenario_name]

    def generate_fn(prefix_frames, n_frames):
        import modal

        requested = requested_native_frames(n_frames, target_fps)
        f = modal.Function.from_name(app_name, function_name)
        if conditioning == "image":
            last_frame = np.asarray(prefix_frames[-1])   # single-image conditioning -- see module docstring
            raw = np.asarray(f.remote(image=last_frame, prompt=prompt, seed=seed, num_frames=requested,
                                      num_inference_steps=num_inference_steps, guidance_scale=guidance_scale))
            return resample_to_target(raw, COSMOS3_NATIVE_FPS, target_fps, n_frames, target_hw)
        prefix = np.asarray(prefix_frames)
        n_native_prefix = max(COSMOS3_V2V_CONDITION_FRAMES,
                              int(np.floor((prefix.shape[0] - 1) * COSMOS3_NATIVE_FPS / target_fps)) + 1)
        cond = resample_to_target(prefix, target_fps, COSMOS3_NATIVE_FPS, n_native_prefix,
                                  (prefix.shape[1], prefix.shape[2]))[-COSMOS3_V2V_CONDITION_FRAMES:]
        total = COSMOS3_V2V_CONDITION_FRAMES + requested
        raw = np.asarray(f.remote(video=cond, prompt=prompt, seed=seed, num_frames=total,
                                  n_condition_frames=COSMOS3_V2V_CONDITION_FRAMES,
                                  num_inference_steps=num_inference_steps, guidance_scale=guidance_scale))
        generated = raw[COSMOS3_V2V_CONDITION_FRAMES:]
        return resample_to_target(generated, COSMOS3_NATIVE_FPS, target_fps, n_frames, target_hw)

    return generate_fn
