# MC-PhysBench

**How long a video world model's prediction of a physical scene stays physically valid, and what breaks first.**

Report & leaderboard: https://www.midcentury.xyz/blog/mc-physbench

MC-PhysBench is a measuring instrument, not a similarity score. Each model is shown one rendered frame of a simple simulated scene and asked to continue it. Its pixels are tracked back into object state and compared, frame by frame, against a reference ensemble of physically valid continuations. The first sustained departure from the ensemble's tolerance band ends the episode, and the detector that caught it names the cause. Fifty paired episodes per model and scene give a survival curve; models are ranked per scene by restricted mean validity time with paired bootstrap tests. The overall score on the leaderboard is a mean across scenes for readability; the per-scene rankings are the measurement.

The Python package is importable as `mcphysbench`; environment variables use the `MCPB_` prefix.

---

## The core loop

```
M reference rollouts from perturbed initial conditions, rendered and tracked
  -> threshold per channel = 99th percentile of reference-vs-reference deviation
  -> render a 1 s prefix, hand the model its last frame, score 2 s of continuation
  -> first crossing sustained for 0.3 s ends the episode: (time, channel)
  -> Kaplan-Meier survival curve over 50 paired episodes
  -> RMVT (area under the curve) ranks; VI50 (median) is reported
  -> displayed shares use the generated window: (RMVT - 1 s) / (ceiling RMVT - 1 s)
```

Seven failure channels: R1 existence, R2 interpenetration, R3 kinematic, R4 long-horizon drift (scored independently), R5 frame invariance, R6 conservation and R7 shape (soft bodies). Nine scenes cover object permanence, contact, rigid dynamics and soft-body conservation.

## Setup

Requirements: Python 3.10+, MuJoCo 3.12.0 (pinned: the soft-body scenes refuse to load on 3.13), and a [Modal](https://modal.com) account for GPU work (the tracker and every self-hosted model run there).

```bash
git clone https://github.com/Midcentury-Labs/mc-physbench.git
cd mc-physbench
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[physics,viz]" "mujoco==3.12.0" modal imageio[ffmpeg] pillow matplotlib
modal setup                                   # once, links your Modal account

python3 tests/test_core.py                    # no MuJoCo needed
python3 -m pytest -q tests                    # full suite, ~1 min

modal deploy remote/modal_app.py              # the tracker ("vitals-phi": SAM2 + DINOv2)
modal deploy remote/modal_app_orchestrate.py  # runs populations on Modal, writes to the results Volume
```

Secrets, only for the providers you intend to run: `vitals-runway` (`RUNWAYML_API_SECRET`) for the Runway gateway models, `vitals-cosmos` (a Hugging Face token with access to NVIDIA's gated guardrail repo) for Cosmos. Create them with `modal secret create <name> KEY=value`.

A quick end-to-end check that needs no GPU:

```bash
MCPB_MANIFEST=configs/manifests/ramp_descent.yaml MCPB_BACKEND=mujoco \
  python3 scripts/run_eval.py --adapter constant_velocity --n-episodes 10
```

## Run your own model

The whole interface is one function:

```python
def generate_fn(prefix_frames: np.ndarray, n_frames: int) -> np.ndarray:
    """prefix_frames: (30, 240, 320, 3) uint8, the rendered 1 s prefix at 30 fps.
    Returns (n_frames, 240, 320, 3) uint8: the model's continuation, already
    resampled to 30 fps and 320x240 (mcphysbench.adapters.video_utils.resample_to_target)."""
```

Steps, each a few lines (see `ADDING_A_MODEL.md` for the worked checklist and the pitfalls already hit):

1. Write `mcphysbench/adapters/<model>.py` exposing `make_<model>_generate_fn(...)`. Use the frozen per-scene prompt from `scenario_prompts.SCENARIO_PROMPTS`; never invent one at call time. If the model is self-hosted, add `remote/modal_app_<model>.py` with its own cache Volume (`mcphysbench/adapters/kandinsky.py` and `remote/modal_app_kandinsky.py` are the smallest complete example; `mcphysbench/adapters/runway.py` and `openai_sora.py` show an API-hosted one).
2. Register it in `mcphysbench/adapters/__init__.py::MODEL_REGISTRY`, with `restricted=True` and a `restriction_note` if its licence has a territorial, scale or commercial clause. Restricted models run only with `MCPB_ENABLE_RESTRICTED_MODELS=<model>` (or `--enable-restricted=<model>` on an orchestrator submit).
3. Add one dispatch line in `scripts/run_model_population.py::make_generate_fn()` and a per-episode timeout in `EPISODE_TIMEOUT_S_DEFAULT`.
4. Smoke it on a few seeds before anything else:

```bash
python3 scripts/run_model_population.py --model <model> --scenario collision --seeds 1,2,3
```

Then run the full cell. Every model is scored on the same 50 episodes (seeds 1 to 50), with frames saved so the frame-invariance channel can be re-scored:

```bash
python3 remote/call_orchestrate.py submit --model <model> --scenario <scene> \
    --seeds $(seq -s, 1 50) --save-frames --n-video-episodes 4
python3 remote/call_orchestrate.py status <call id>
python3 remote/call_orchestrate.py pull --only <scene>_<model>
```

`remote/call_chain.py` queues many cells on Modal itself so a laptop going to sleep cannot stall them. A cell with fewer than 50 episodes is shown greyed and unranked; top it up with `--seeds <missing> --merge-existing`.

## Reproduce the leaderboard

Every number on the site comes from `results/leaderboard.json`, which is built from the population files in `results/`. To regenerate from scratch for one scene:

```bash
S=occlusion_reemergence
M=configs/manifests/$S.yaml

# 1. Instrument validation (once per scene): planted defects on state, then through video
MCPB_MANIFEST=$M MCPB_BACKEND=mujoco python3 scripts/run_l0_demo.py      # GATE 1
MCPB_MANIFEST=$M MCPB_BACKEND=mujoco python3 scripts/run_gate2.py        # GATE 2, needs vitals-phi

# 2. Baselines and the instrument ceiling, on the same seeds as the models
MCPB_MANIFEST=$M MCPB_BACKEND=mujoco python3 scripts/run_eval.py --adapter constant_velocity --n-episodes 50 --seed-base 1
MCPB_MANIFEST=$M MCPB_BACKEND=mujoco python3 scripts/run_eval.py --adapter copy_last_state   --n-episodes 50 --seed-base 1
python3 remote/call_orchestrate.py submit --model truth_render --scenario $S --seeds $(seq -s, 1 50) --save-frames

# 3. Every model (see above), one submit per model and scene

# 4. Band re-score, ranking, pages
python3 scripts/leaderboard_stability.py        # re-scores every cell at 0.5x / 1x / 2x the reference band
python3 scripts/render_leaderboard.py           # -> results/leaderboard.json + results/pages/leaderboard.html
                                                 #    shares and the overall score are on the generated window (see share_of_ceiling)
python3 scripts/render_score_report.py          # PDF report, population pages, index; add --deploy to publish
```

The ranking rules were written before the full matrix existed and are reproduced verbatim on the leaderboard page; where a rule was amended, the original text stays visible with the reason. Thresholds for the frame-invariance channel are calibrated on the renderer that produced the frames, so a cell must be scored on the machine that generated it, which is why populations run on the orchestrator.

## Layout

```
mcphysbench/            the library: physics, mutants, detectors, the tracker (phi/), survival stats, adapters
scenes/            one MJCF file per scene
configs/manifests/ one EpisodeSpec per scene: the source of truth for every scene parameter
scripts/           entry points: run_eval, run_model_population, run_gate2, leaderboard_stability,
                   render_leaderboard, render_score_report, render_population_pages
remote/            Modal apps (one per model, the tracker, the orchestrator, the chain runner) and their call_*.py clients
tests/             one file per module; each runs standalone or under pytest
results/           generated: population JSON, trajectories, frames, videos, pages. Never edited by hand
AGENT.md           the design log: every milestone, decision and finding, in the order it happened
```

## Citation

```bibtex
@misc{shah2026mcphysbench,
  title        = {MC-PhysBench: A Time-to-Failure Physics Benchmark for World Models},
  author       = {Shah, Shilpi},
  year         = {2026},
  publisher    = {Midcentury},
  howpublished = {\url{https://github.com/Midcentury-Labs/mc-physbench}},
  note         = {Report and leaderboard at \url{https://www.midcentury.xyz/blog/mc-physbench}}
}
```

## License

Apache License 2.0. See `LICENSE`. Model weights and hosted APIs evaluated by the benchmark carry their own licences; the registry records each one, and restricted models are off by default.
