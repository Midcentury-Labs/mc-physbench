"""Modal app for Kandinsky WM 1.0 (Sber / Kandinsky Lab), the General
Physics image-to-video checkpoint -- backs `mcphysbench/adapters/kandinsky.py`'s
own `generate_fn`. Its own app, its own cache Volume, same "one small file
per model" discipline as every other model under `remote/`.

Model (verified on the Hub 2026-10-05, not assumed):
`kandinskylab/Kandinsky-WM-1.0-I2V-5s-PH` -- MIT license, not gated, a
one-line diffusers pipeline (`Kandinsky5I2VPipeline`; model_index.json
names Kandinsky5Transformer3DModel + AutoencoderKLHunyuanVideo +
Qwen2.5-VL + CLIP). The family has three domain checkpoints (AV, RO, PH);
PH = general physics is the one that matches this benchmark. 2B DiT on
the Kandinsky 5.0 Video Lite base, RL post-trained (GRPO) against a
physical-plausibility reward. Generates 121 frames = 5 s at 768x512, the
card exports at fps=24.

**GPU tier: A100-80GB.** The DiT is small (2B) but the text encoder is
Qwen2.5-VL-7B (five safetensors shards), the same co-residency that
OOM'd HunyuanVideo-1.5 on a 24 GB A10G the same day. Not worth repeating
that experiment; the first real call is still a debugging pass for
everything else (timing, output shape, frame rate).

**Prompting.** The SAME frozen `scenario_prompts.SCENARIO_PROMPTS` text
as every other model; the model card's own default negative prompt is
passed verbatim as a model-native default (the same kind of decision as
Cosmos's JSON container), documented here, never a per-scenario rewrite.

Usage:
    modal deploy remote/modal_app_kandinsky.py
    python3 remote/call_kandinsky.py download_checkpoints
    python3 remote/call_kandinsky.py image2video --input-path prefix_last_frame.png --prompt "..." --save-path out.mp4
"""
import modal

GPU_TYPE = "A100-80GB"
MODEL_ID = "kandinskylab/Kandinsky-WM-1.0-I2V-5s-PH"
# The model card's own default negative prompt (Path B quickstart), verbatim.
NEGATIVE_PROMPT = ("Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
                   "worst quality, low quality, ugly, deformed, walking backwards")
NATIVE_H, NATIVE_W, NATIVE_FRAMES = 512, 768, 121

app = modal.App("vitals-kandinsky")

kandinsky_cache = modal.Volume.from_name("vitals-kandinsky-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "torchvision",         # Qwen2VLVideoProcessor (the text encoder's processor) imports it -- first call's ImportError, 2026-10-05
        "diffusers>=0.40.0",   # Kandinsky5I2VPipeline; 0.40.0 is on PyPI (same pin as vitals-hunyuan)
        "transformers>=4.45",
        "accelerate",
        "sentencepiece",
        "ftfy",
        "pillow",
        "imageio-ffmpeg",
    )
    .env({"HF_HOME": "/cache/hf"})
)


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": kandinsky_cache}, timeout=3600)
def download_checkpoints():
    """One-time: pull the weights into the persistent Volume so the first
    real call does not pay the ~20 GB download inside its own timeout."""
    import torch
    from diffusers import Kandinsky5I2VPipeline

    Kandinsky5I2VPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    kandinsky_cache.commit()
    return "CHECKPOINTS_DOWNLOADED"


@app.function(image=image, gpu=GPU_TYPE, volumes={"/cache": kandinsky_cache}, timeout=1800)
def image2video(image, prompt: str, seed: int = 0, num_frames: int = NATIVE_FRAMES,
                num_inference_steps: int = 50, guidance_scale: float = 5.0):
    """ONE Kandinsky WM 1.0 image-to-video generation -> (T,H,W,3) uint8.
    `image` is a plain (H,W,3) uint8 array; PIL conversion happens here so
    the caller never needs torch/diffusers. Height/width are the model's
    own native 512x768; the card's defaults for steps (50) and guidance
    (5.0) are kept."""
    import numpy as np
    import torch
    from diffusers import Kandinsky5I2VPipeline
    from PIL import Image

    pipe = Kandinsky5I2VPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    pil_image = Image.fromarray(np.asarray(image)).convert("RGB")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(
        image=pil_image, prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
        height=NATIVE_H, width=NATIVE_W, num_frames=num_frames,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        generator=generator, output_type="np",
    ).frames[0]

    kandinsky_cache.commit()
    frames = np.asarray(output)
    if frames.dtype != np.uint8:          # (T,H,W,3) float in [0,1] -> uint8
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    return frames
