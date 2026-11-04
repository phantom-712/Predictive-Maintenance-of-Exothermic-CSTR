# Predictive Maintenance of an Exothermic CSTR

This repository models a thermal runaway risk problem in an exothermic continuous stirred-tank reactor (CSTR) and frames it as a predictive maintenance and remaining-safe-time estimation task. The workflow spans synthetic data generation, time-series forecasting with an LSTM, and safety-oriented prognosis evaluation.

## Live demo

Try the [CSTR Thermal Runaway Inference app](https://umid5-cstr-thermal-runaway-inference.static.hf.space/index.html). It runs an interactive CSTR simulation in the browser, with stable and upset scenarios and an LSTM estimate of remaining time to cutoff. Simulation data stays in the browser. The model is trained on synthetic data and is for benchmarking only; never use its output to operate or make safety decisions for a real reactor.

## Project objective

The reactor is governed by an Arrhenius reaction term and a thermal energy balance. Under normal operation, the system remains stable, but a set of upset scenarios such as coolant pump failure, loss of jacket cooling, feed surge, and overheating can drive the temperature toward a runaway threshold. The project aims to:

- generate realistic transient reactor trajectories,
- identify when a trajectory is approaching a critical cutoff,
- estimate the remaining time to runaway,
- evaluate model quality using safety and prognostic metrics,
- analyse sensor noise robustness and demonstrate the necessity of a signal filtering pipeline.

## Repository structure

```text
.
├── README.md
├── dataset/
│   └── cstr_21k_dataset_parallel.csv
├── notebook/
│   └── Untitled4.ipynb
├── plots/
│   ├── 01_input_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_safety_analysis.png
│   ├── 04_noise_robustness.png
│   └── cstr_plots.png
├── scripts/
│   ├── task_1.py
│   ├── task_2.py
│   ├── task_3.py
│   ├── task_4_noise_robustness.py
│   └── cstr_runaway_trajectories.csv
├── huggingface_static_space/
│   ├── index.html          ← updated: Sensor Noise toggle
│   ├── model.onnx
│   ├── scaler.json
│   └── default_profiles.json
```

## Data and reactor model

The system dynamics are represented by the state variables:

- `cA`: reactor concentration
- `T`: reactor temperature
- `q`: flow rate
- `cAi`: inlet concentration
- `Ti`: inlet temperature
- `Tc`: coolant temperature
- `dT_dt`: temperature derivative

The core reaction is modeled with:

- Arrhenius kinetics: $k = k_0 \exp(-E/R T)$
- concentration balance: $dcA/dt = (q/V)(cAi - cA) - r_A$
- energy balance: $dT/dt = (q/V)(Ti - T) + ((-\Delta H)/\rho C_p) r_A - (UA/(V \rho C_p))(T - Tc)$

The simulation includes a safety cutoff at 380 K and a warning window of up to 20 minutes. Each trajectory is labelled with:

- `time_to_cutoff`
- `is_runaway`
- `run_id`

## Script workflow

### 1) `scripts/task_1.py`

This script simulates many CSTR runs under upset conditions using `scipy.integrate.solve_ivp` and creates the synthetic operational dataset.

Key behaviors:

- initializes random reactor conditions for each run,
- injects disturbances such as cooling failure, feed surge, and temperature excursions,
- stops the simulation when the temperature reaches the cutoff threshold,
- computes `time_to_cutoff` and `is_runaway` labels,
- writes the output to `scripts/cstr_runaway_trajectories.csv`.

Example (from the `scripts/` directory, which matches the CSV output path used in the code):

```bash
cd scripts
python task_1.py
```

### 2) `scripts/task_2.py`

This script performs a sliding-window LSTM inference check on a selected trajectory.

What it does:

- loads the generated CSTR trajectory CSV,
- recreates the training scaler from the training split,
- feeds the past 15 points into an LSTM model,
- predicts the remaining time to cutoff,
- overlays actual vs predicted time-to-cutoff and reactor temperature on a plot.

Important note:

- The script assumes a saved model checkpoint at a path like `/content/cstr_lstm_model.pth`.
- If you are running locally, update the `CSV_FILE` and `MODEL_PATH` constants to match your environment.

Example for a Colab-style environment (or after updating the constants to your local paths):

```bash
python scripts/task_2.py
```

### 3) `scripts/task_3.py`

This script evaluates the prognostic quality of a time-to-cutoff prediction model.

It computes:

- global RMSE and MAE,
- critical-zone RMSE and MAE for predictions within the danger window,
- PHM-style asymmetric prognostic penalty score,
- mean lead time,
- missed runaway count,
- false alarm rate.

It can also generate a trajectory-level evaluation plot comparing ground truth and predicted time to cutoff.

Example (run from the directory containing the generated CSV, or update the script to point to your chosen file):

```bash
python scripts/task_3.py
```

### 4) `scripts/task_4_noise_robustness.py`

This script demonstrates the necessity of a signal filtering pipeline by analysing the impact of sensor noise on CSTR temperature monitoring.

What it does:

- loads a sample runaway trajectory from `scripts/cstr_runaway_trajectories.csv`,
- selects the first runaway run as a representative case,
- injects 5% Gaussian noise (`Normal(0, 0.05 × T)`) into the reactor temperature `T` and coolant temperature `Tc` columns,
- applies a simple rolling-window moving average filter to show signal reconstruction,
- counts premature false trip alarms caused by raw noise crossing the 380 K safety threshold before the reactor is physically unsafe,
- saves a comparison plot of clean vs. noisy vs. filtered temperature traces to `plots/04_noise_robustness.png`.

Key finding: at a 5% noise level (σ ≈ 17.5 K at 350 K), raw sensor noise can generate false safety trip alarms several time-steps before an actual runaway, motivating pre-processing and filtering before LSTM inference.

Example:

```bash
python scripts/task_4_noise_robustness.py
```

Output saved to `plots/04_noise_robustness.png`.

## Included datasets

### `dataset/cstr_21k_dataset_parallel.csv`

This file contains a larger synthetic reactor dataset with thousands of observations and reactor operating variables. It is useful for exploratory data analysis and model development.

### `scripts/cstr_runaway_trajectories.csv`

This file contains the event-centric trajectory dataset generated in `task_1.py` for dynamic run-to-run analysis and time-to-cutoff prediction.

## Generated analysis outputs

The `plots/` directory contains exploratory visuals such as:

- input distributions,
- feature correlation heatmaps,
- safety analysis plots,
- example trajectory evaluations,
- sensor noise robustness comparison (clean vs. noisy vs. filtered temperature traces).

## Environment setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run the scripts in this order, keeping the CSV in the same working directory as the code expects:

```bash
cd scripts
python task_1.py
python task_2.py
python task_3.py
python task_4_noise_robustness.py
```

## Interactive frontend — Sensor Noise toggle

The browser demo at `huggingface_static_space/index.html` now includes a **Simulate Sensor Noise** toggle switch in the Operating inputs panel. When enabled, the JavaScript simulation loop injects 5% Gaussian noise (using a Box-Muller transform for a physically realistic normal distribution) into the real-time `T` and `Tc` readings plotted on the chart. This visually demonstrates to users why a data curation and filtering pipeline is critical before feeding raw sensor signals into the LSTM. The physical state integrator continues to evolve using the true (noiseless) values so the simulation is not prematurely terminated by a measurement spike, and any noise-induced crossing of the 380 K threshold is flagged separately as a **Sensor Trip Alarm (Noise Spike)** in the status badge.

## Interpretation

This project is designed as a synthetic benchmark for predictive maintenance of safety-critical process systems. It demonstrates how an LSTM can estimate remaining useful time before runaway develops and how safety metrics can be used to assess whether those forecasts are operationally useful.

## Notes

- The project uses synthetic data rather than real plant telemetry.
- `task_2.py` is model-dependent and requires a trained LSTM checkpoint.
- `task_3.py` can generate demo predictions if no `y_pred` column is present, which is helpful for testing the evaluation workflow.
- `task_4_noise_robustness.py` reads `scripts/cstr_runaway_trajectories.csv` and writes the output plot to `plots/04_noise_robustness.png`. Run `task_1.py` first if the CSV does not exist.

