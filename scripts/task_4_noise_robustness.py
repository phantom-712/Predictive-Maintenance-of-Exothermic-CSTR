"""
Task 4: Noise Robustness and Signal Integrity Analysis

This script demonstrates sensor noise injection and evaluates its impact
on CSTR temperature monitoring against the 380 K critical safety cutoff threshold.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. Configuration & Path Handling
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Resolve trajectory dataset path
CSV_PATH_SCRIPTS = SCRIPT_DIR / "cstr_runaway_trajectories.csv"
CSV_PATH_DATASET = REPO_ROOT / "dataset" / "cstr_21k_dataset_parallel.csv"

if CSV_PATH_SCRIPTS.exists():
    DATA_PATH = CSV_PATH_SCRIPTS
elif CSV_PATH_DATASET.exists():
    DATA_PATH = CSV_PATH_DATASET
else:
    raise FileNotFoundError("Trajectory CSV not found in scripts/ or dataset/ directory.")

OUTPUT_PLOT_DIR = REPO_ROOT / "plots"
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT_PATH = OUTPUT_PLOT_DIR / "04_noise_robustness.png"

# Safety Parameters
T_CUTOFF = 380.0  # [K] Safety Trip Threshold
NOISE_STD_RATIO = 0.05  # 5% Gaussian noise

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 2. Data Loading & Runaway Selection
# ==========================================
print(f"Loading trajectory data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Select a sample runaway run
runaway_runs = df[df["is_runaway"] == 1]["run_id"].unique()
if len(runaway_runs) == 0:
    raise ValueError("No runaway trajectories found in the dataset.")

# Select first runaway run (e.g., run_id = 2)
selected_run_id = runaway_runs[0]
run_df = df[df["run_id"] == selected_run_id].copy().sort_values("time")
print(f"Selected sample runaway run: Run ID {selected_run_id} ({len(run_df)} time steps)")

# ==========================================
# 3. Inject 5% Gaussian Noise
# ==========================================
# Sensor noise injection: Normal(mean=0, std=0.05 * value)
noise_T = np.random.normal(0, NOISE_STD_RATIO * run_df["T"].values)
noise_Tc = np.random.normal(0, NOISE_STD_RATIO * run_df["Tc"].values)

run_df["T_noisy"] = run_df["T"].values + noise_T
run_df["Tc_noisy"] = run_df["Tc"].values + noise_Tc

# Optional baseline filter (Rolling Window / Moving Average) to demonstrate filtering benefit
run_df["T_filtered"] = run_df["T_noisy"].rolling(window=5, min_periods=1, center=True).mean()

# Analysis of noise effects
false_alarms = ((run_df["T_noisy"] >= T_CUTOFF) & (run_df["T"] < T_CUTOFF)).sum()
print(f"Clean Temp Range:  {run_df['T'].min():.2f} K to {run_df['T'].max():.2f} K")
print(f"Noisy Temp Range:  {run_df['T_noisy'].min():.2f} K to {run_df['T_noisy'].max():.2f} K")
print(f"Premature false alarms caused by raw noise (T_noisy >= {T_CUTOFF} K): {false_alarms} time-steps")

# ==========================================
# 4. Plotting
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot Clean vs. Noisy Temperature against Time-to-Cutoff
ax.plot(
    run_df["time_to_cutoff"],
    run_df["T"],
    color="#1f77b4",
    linewidth=2.5,
    label="Clean Temperature (T)",
)
ax.plot(
    run_df["time_to_cutoff"],
    run_df["T_noisy"],
    color="#d62728",
    alpha=0.6,
    linestyle="-",
    linewidth=1.2,
    label=f"Noisy Temperature (T + 5% Gaussian Noise)",
)
ax.plot(
    run_df["time_to_cutoff"],
    run_df["T_filtered"],
    color="#2ca02c",
    linestyle="--",
    linewidth=1.8,
    label="Filtered Signal (Moving Average, w=5)",
)

# 380 K Cutoff threshold
ax.axhline(
    y=T_CUTOFF,
    color="black",
    linestyle=":",
    linewidth=2,
    label=f"Safety Cutoff Threshold ({T_CUTOFF:.0f} K)",
)

# Invert X-axis so time naturally progresses towards cutoff (t_cutoff = 0 on right)
ax.invert_xaxis()

# Styling
ax.set_title(
    f"Sensor Noise Robustness & Signal Integrity: Run {selected_run_id}\n(Clean vs. Noisy vs. Filtered Temperature)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("Time to Cutoff (minutes remaining) [-> Approaching Runaway]", fontsize=11)
ax.set_ylabel("Reactor Temperature (K)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH)
plt.close()

print(f"Plot successfully saved to: {OUTPUT_PLOT_PATH}")
