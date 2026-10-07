"""adapters/cosmos3.py -- Cosmos 3 Nano generate_fn. Same honest boundary
as test_wan_adapter.py: covers everything except the live Modal call."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest


def test_requires_a_registered_scenario_prompt():
    from mcphysbench.adapters import cosmos3
    with pytest.raises(KeyError):
        cosmos3.make_cosmos3_generate_fn_modal("not_a_real_scenario")


def test_shares_the_identical_prompt_table_and_resampler():
    """Matched settings: same frozen prompt object and the same shared
    resampler as every other model -- no local overrides that could drift."""
    from mcphysbench.adapters import cosmos3, wan, video_utils
    assert cosmos3.SCENARIO_PROMPTS is wan.SCENARIO_PROMPTS
    assert cosmos3.resample_to_target is video_utils.resample_to_target


def test_requested_native_frames_just_cover_n_frames_and_respect_the_cap():
    """24 fps native vs 30 fps target: request just enough (+1), never the
    full 189 by default -- that's ~4x the H100 time for frames that get
    trimmed anyway."""
    from mcphysbench.adapters.cosmos3 import COSMOS3_NATIVE_FPS, COSMOS3_MAX_FRAMES
    def requested(n_frames, target_fps=30):
        return min(COSMOS3_MAX_FRAMES, max(5, int(np.ceil(n_frames * COSMOS3_NATIVE_FPS / target_fps)) + 1))
    assert COSMOS3_NATIVE_FPS == 24
    assert requested(60) == 49            # 2.0s -> 48 native + 1
    assert requested(120) == 97           # 4.0s
    assert requested(1) == 5              # floor
    assert requested(10_000) == COSMOS3_MAX_FRAMES


def test_registered_as_an_unrestricted_real_backend():
    from mcphysbench.adapters import MODEL_REGISTRY, REAL_MODEL_BACKENDS
    assert MODEL_REGISTRY["cosmos3nano"]["restricted"] is False
    assert "cosmos3nano" in REAL_MODEL_BACKENDS


def test_video_conditioned_arm_discards_the_conditioning_span_and_keeps_the_contract():
    """cosmos3nano_v2v / cosmos3super_v2v (2026-09-25): the prefix's last
    21 native frames (6 latents, 4k+1) are sent as `video=`, the app returns
    conditioning + generated, the adapter drops the first 21 and resamples
    the remainder to exactly n_frames at 30 fps -- the same continuation
    contract as the image path. Exercised with a fake modal Function."""
    import types, sys
    from mcphysbench.adapters import cosmos3
    calls = {}

    class FakeFn:
        def remote(self, **kw):
            calls.update(kw)
            n = kw["num_frames"]
            return np.full((n, 8, 8, 3), 7, np.uint8)
    fake_modal = types.SimpleNamespace(Function=types.SimpleNamespace(from_name=lambda app, fn: (calls.__setitem__("fn", fn), FakeFn())[1]))
    sys.modules["modal"] = fake_modal
    try:
        gen = cosmos3.make_cosmos3_generate_fn_modal("collision", conditioning="video", target_hw=(8, 8))
        prefix = np.zeros((30, 8, 8, 3), np.uint8)
        out = gen(prefix, 60)
    finally:
        del sys.modules["modal"]
    assert calls["fn"] == "video2video"
    assert calls["video"].shape[0] == cosmos3.COSMOS3_V2V_CONDITION_FRAMES == 21
    assert (cosmos3.COSMOS3_V2V_CONDITION_FRAMES - 1) % 4 == 0
    assert calls["n_condition_frames"] == 21
    assert calls["num_frames"] == 21 + cosmos3.requested_native_frames(60)   # total clip, conditioning + generated
    assert out.shape == (60, 8, 8, 3)


def test_video_arms_are_registered_as_separate_unrestricted_backends():
    from mcphysbench.adapters import MODEL_REGISTRY, REAL_MODEL_BACKENDS
    for name in ("cosmos3nano_v2v", "cosmos3super_v2v"):
        assert MODEL_REGISTRY[name]["restricted"] is False
        assert MODEL_REGISTRY[name]["conditioning"] == "video"
        assert name in REAL_MODEL_BACKENDS
    assert "conditioning" not in MODEL_REGISTRY["cosmos3nano"]
