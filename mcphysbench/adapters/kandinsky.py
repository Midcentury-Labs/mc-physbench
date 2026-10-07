"""generate_fn for Kandinsky WM 1.0, General Physics checkpoint (Sber /
Kandinsky Lab, 2026-10) -- plugs into `VideoWorldModel`'s `generate_fn`
slot via the deployed `vitals-kandinsky` Modal app
(remote/modal_app_kandinsky.py).

https://huggingface.co/kandinskylab/Kandinsky-WM-1.0-I2V-5s-PH

Facts this adapter is built on (read off the model card and
model_index.json on 2026-10-05):
- MIT license, not gated -> registered `restricted=False`.
- Single-IMAGE conditioning (`Kandinsky5I2VPipeline` takes one `image`),
  so only the prefix's last frame reaches the model, like wan/hunyuan.
- Text prompt required -> the SAME frozen `SCENARIO_PROMPTS`; the card's
  own default negative prompt is passed inside the Modal app as a
  model-native default (documented there).
- Native output: 121 frames = 5 s at 768x512; the card exports at
  fps=24, so NATIVE_FPS = 24 is the card's own number, not a guess.
  Like every other model, `resample_to_target` trims/resamples to the
  2 s the protocol scores.
- The DiT is 2B (Kandinsky 5.0 Video *Lite* base). The world-model
  family has no larger size, so it passes the flagship-tier rule as a
  family; the user was told a 19B Kandinsky 5.0 Pro I2V exists.
"""
import numpy as np

from .scenario_prompts import SCENARIO_PROMPTS
from .video_utils import resample_to_target

KANDINSKY_NATIVE_FPS = 24
KANDINSKY_NATIVE_FRAMES = 121


def make_kandinsky_generate_fn_modal(scenario_name, app_name="vitals-kandinsky", function_name="image2video",
                                     target_fps=30, target_hw=(240, 320), seed=0,
                                     num_inference_steps=50):
    """Builds `generate_fn(prefix_frames, n_frames) -> continuation_frames`
    calling the deployed `vitals-kandinsky` Modal app. Needs
    `modal deploy remote/modal_app_kandinsky.py` first."""
    if scenario_name not in SCENARIO_PROMPTS:
        raise KeyError(f"no frozen prompt for scenario {scenario_name!r} -- add one to "
                        f"scenario_prompts.SCENARIO_PROMPTS, do not invent one at call time "
                        f"(pre-registration).")
    prompt = SCENARIO_PROMPTS[scenario_name]

    def generate_fn(prefix_frames, n_frames):
        import modal

        last_frame = np.asarray(prefix_frames[-1])   # single-image conditioning
        # The pipeline's frame count follows the VAE's temporal grid (4k+1);
        # asking for the native 121 keeps the call on the card's own path
        # and lets resample_to_target trim. A shorter count is NOT requested
        # here until a real call shows the pipeline honours it -- the first
        # call is the debugging pass for that.
        f = modal.Function.from_name(app_name, function_name)
        raw_frames = f.remote(image=last_frame, prompt=prompt, seed=seed,
                              num_frames=KANDINSKY_NATIVE_FRAMES,
                              num_inference_steps=num_inference_steps)
        raw_frames = np.asarray(raw_frames)
        return resample_to_target(raw_frames, KANDINSKY_NATIVE_FPS, target_fps, n_frames, target_hw)

    return generate_fn
