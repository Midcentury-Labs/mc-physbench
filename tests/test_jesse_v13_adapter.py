"""mcphysbench/adapters/jesse_v13.py -- API-access world model adapter tests.
No API key or network required: injects a fake client via `client_factory` and
writes a real local mp4 file so the decode and resample path is exercised end-to-end."""
import base64
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from mcphysbench.adapters import jesse_v13 as jv
from mcphysbench.adapters.video_utils import _write_video


class _FakeSolidSFClient:
    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    def rollout(self, **kwargs):
        self._calls.append(kwargs)
        return self._response


def _fake_video_url(tmp_path, n_frames, fps, hw=(120, 160)):
    frames = (np.random.default_rng(0).integers(0, 255, size=(n_frames, *hw, 3))).astype(np.uint8)
    p = tmp_path / "jesse_gen.mp4"
    _write_video(frames, p, fps=fps)
    return p.as_uri()


def test_frame_to_data_uri_is_lossless_png():
    frame = np.random.default_rng(1).integers(0, 255, size=(8, 10, 3)).astype(np.uint8)
    uri = jv.frame_to_data_uri(frame)
    assert uri.startswith("data:image/png;base64,")
    from PIL import Image

    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))
    assert np.array_equal(decoded, frame), "PNG must round-trip the exact rendered pixels"


def test_generate_fn_sends_matched_conditioning_and_returns_exact_frames(tmp_path):
    calls = []
    url = _fake_video_url(tmp_path, n_frames=60, fps=30)
    client = _FakeSolidSFClient(
        {"id": "rollout_123", "status": "succeeded", "video_url": url, "cost_credits": 12},
        calls,
    )
    logs = []
    gen = jv.make_jesse_generate_fn(
        "collision",
        model="jesse-1.3-swarm-100k",
        seed=0,
        client_factory=lambda: client,
        log=logs.append,
    )

    prefix = np.zeros((30, 240, 320, 3), dtype=np.uint8)
    prefix[-1, 5, 7] = (200, 30, 30)
    out = gen(prefix, n_frames=60)

    assert out.shape == (60, 240, 320, 3) and out.dtype == np.uint8
    assert len(calls) == 1
    kw = calls[0]
    assert kw["model"] == "jesse-1.3-swarm-100k"
    assert kw["n_frames"] == 60 and kw["fps"] == 30
    from mcphysbench.adapters.scenario_prompts import SCENARIO_PROMPTS

    assert kw["prompt_text"] == SCENARIO_PROMPTS["collision"]
    assert kw["prompt_image"] == jv.frame_to_data_uri(prefix[-1])
    assert any("cost_credits=12" in l for l in logs)


def test_generate_fn_raises_on_failed_rollout():
    calls = []
    client = _FakeSolidSFClient({"status": "failed", "error": "upstream timeout"}, calls)
    gen = jv.make_jesse_generate_fn("collision", client_factory=lambda: client)
    with pytest.raises(RuntimeError):
        gen(np.zeros((5, 240, 320, 3), dtype=np.uint8), n_frames=60)


def test_generate_fn_rejects_unregistered_scenario_prompt():
    with pytest.raises(KeyError):
        jv.make_jesse_generate_fn("no_such_scenario", client_factory=lambda: None)


def test_registry_exposes_jesse_models_as_real_api_backends():
    from mcphysbench.adapters import MODEL_REGISTRY, REAL_MODEL_BACKENDS

    for name in ("jesse_1_3_single", "jesse_1_3_swarm_100k"):
        assert MODEL_REGISTRY[name]["access"] == "api"
        assert MODEL_REGISTRY[name]["provider"] == "solidsf"
        assert MODEL_REGISTRY[name]["conditioning"] == "image"
        assert name in REAL_MODEL_BACKENDS
