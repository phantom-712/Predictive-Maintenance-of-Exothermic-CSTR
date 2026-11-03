---
title: CSTR Thermal Runaway Inference
emoji: ⚠️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# CSTR Thermal Runaway Inference

This Gradio app predicts the synthetic CSTR model's remaining time to thermal cutoff.

Upload a single time-ordered CSV trajectory with at least 15 rows. It must include
`q`, `cAi`, `Ti`, `Tc`, `cA`, `T`, and `dT_dt`; `time` is optional.

`example_trajectory.csv` is a bundled synthetic runaway trajectory you can use to
try the interface immediately.

The output is a model estimate, intended solely for this synthetic benchmark. It must
not be used to operate or make safety decisions for a real reactor.
