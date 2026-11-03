---
title: CSTR Thermal Runaway Inference
emoji: ⚠️
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: Browser-based CSTR remaining-safe-time prediction
---

# CSTR Thermal Runaway Inference

This free Static Space runs an interactive CSTR simulation entirely in the
browser. It includes ready-to-run stable and cooling/feed-upset presets, then
uses the four live operating controls as inputs to the reactor equations:

- flow rate (`q`)
- inlet concentration (`cAi`)
- inlet temperature (`Ti`)
- coolant temperature (`Tc`)

The graph begins empty. Set a duration, choose a Task 1 operating mode, then
choose **Run CSTR simulation**. The four upset modes match `task_1.py`:
coolant pump failure, jacket loss, feed surge, and overheating. The random mode
uses Task 1's 75% upset probability, random 4–10 minute trigger time, and a
random upset scenario. Each run advances the reactor mass and energy balances
in 0.2-minute increments and progressively adds calculated traces for `q`,
`cAi`, `Ti`, `Tc`, `cA`, `T`, and `dT_dt`.

Once 15 simulated points are available, the included LSTM estimates remaining
time to cutoff at each step; its white dashed trace, live metric, and table
column update with the simulation. No operating data is sent to a server.

The included model is trained only on synthetic data. Its output is a benchmark
estimate and must never be used to operate or make safety decisions for a real
reactor.
