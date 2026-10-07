# results/

Only the files the leaderboard is built from are tracked here. Everything else the pipeline writes (frames, trajectories, rendered pages, the PDF report, gate reports, smoke and debug outputs, superseded model versions, the render-domain variants) is generated and ignored; `.gitignore` lists the rules.

| File | What it is |
|---|---|
| `l0_demo_<scene>_<series>.json` | One population: 50 paired episodes of `<series>` on `<scene>`, scored. `<series>` is a model, a baseline (`constant_velocity`, `copy_last_state`), the physics ideal (`truth`) or the instrument ceiling (`truth_render`). Holds every episode's event time and channel, the Kaplan–Meier curve, the calibrated thresholds and the frame-invariance statistics. |
| `lambda_rescore_<scene>_<model>.json` | The same cached episodes re-scored at wider and narrower reference bands; feeds the resolution plot on each population page. |
| `leaderboard_stability.json` | Per cell, RMVT at 0.5×, 1× and 2× the manifest band (`scripts/leaderboard_stability.py`); the source of the band-specific flag and the position column. |
| `leaderboard.json` | Every number on the leaderboard page (`scripts/render_leaderboard.py`): per-scene rows with RMVT, validity after the prefix, shares of the ceiling on the generated window, paired deltas, ranks, and the cross-scene summary with the overall score. |
| `phi_refs_soft_ramp.npz` | The tracker-measured reference band for the soft-body scene's conservation and shape channels. |
| `videos/<scene>_<series>_grid.mp4` | Four episodes side by side: the rendered prefix (green border), the continuation (red border), the true object (green marker) and the tracked position (red marker). |

Scenes: `occlusion_reemergence`, `occlusion_corridor`, `occlusion_corridor_interpenetration`, `ramp_descent`, `ramp_descent_high_friction`, `collision`, `billiards`, `block_stack`, `soft_ramp`.

To regenerate any of it, see "Reproduce the leaderboard" in the top-level README. A population file is produced by `scripts/run_model_population.py` (models, the ceiling) or `scripts/run_eval.py` (baselines, the physics ideal); the leaderboard files are rebuilt by `scripts/leaderboard_stability.py` then `scripts/render_leaderboard.py`.
