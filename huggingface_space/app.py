"""Gradio inference app for CSTR thermal-runaway remaining-safe-time prediction."""

from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "cstr_lstm_model.pth"
SCALER_PATH = APP_DIR / "cstr_feature_scaler.joblib"
WINDOW_SIZE = 15
FEATURE_COLS = ["q", "cAi", "Ti", "Tc", "cA", "T", "dT_dt"]


class CSTRRunawayLSTM(nn.Module):
    """The model architecture used during training."""

    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the model and exactly the scaler fitted on the training split."""
    model = CSTRRunawayLSTM(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, joblib.load(SCALER_PATH)


def predict_from_csv(csv_path):
    """Predict remaining safe time for every valid trailing window in one trajectory."""
    if csv_path is None:
        raise gr.Error("Upload a CSV file containing a single time-ordered CSTR trajectory.")

    frame = pd.read_csv(csv_path)
    missing_columns = [column for column in FEATURE_COLS if column not in frame.columns]
    if missing_columns:
        raise gr.Error(f"The CSV is missing required columns: {', '.join(missing_columns)}")
    if "run_id" in frame and frame["run_id"].nunique() != 1:
        raise gr.Error("Upload a single trajectory only; the supplied CSV contains multiple run_id values.")
    if len(frame) < WINDOW_SIZE:
        raise gr.Error(f"At least {WINDOW_SIZE} sequential rows are required for an inference.")
    if "time" in frame and not frame["time"].is_monotonic_increasing:
        raise gr.Error("Trajectory rows must be ordered by ascending time.")

    model, scaler = load_artifacts()
    scaled_features = scaler.transform(frame[FEATURE_COLS])
    windows = np.asarray(
        [scaled_features[index : index + WINDOW_SIZE] for index in range(len(frame) - WINDOW_SIZE + 1)],
        dtype=np.float32,
    )
    with torch.no_grad():
        predictions = model(torch.from_numpy(windows)).squeeze(1).numpy()
    predictions = np.clip(predictions, 0.0, 20.0)

    times = frame["time"].iloc[WINDOW_SIZE - 1 :].to_numpy() if "time" in frame else np.arange(WINDOW_SIZE - 1, len(frame))
    temperatures = frame["T"].iloc[WINDOW_SIZE - 1 :].to_numpy()
    results = pd.DataFrame(
        {
            "time_min": times,
            "temperature_k": temperatures,
            "predicted_time_to_cutoff_min": predictions.round(3),
        }
    )

    latest = float(predictions[-1])
    status = (
        f"### Estimated remaining safe time: **{latest:.2f} minutes**\n"
        "Predictions are estimates from a synthetic-data model and must not be used to operate a real reactor."
    )

    figure, left_axis = plt.subplots(figsize=(9, 4.5))
    left_axis.plot(times, predictions, color="crimson", label="Predicted time to cutoff")
    left_axis.axhline(5.0, color="darkorange", linestyle=":", label="Critical warning threshold")
    left_axis.set_xlabel("Simulation time (min)")
    left_axis.set_ylabel("Remaining safe time (min)", color="crimson")
    left_axis.tick_params(axis="y", labelcolor="crimson")
    left_axis.grid(alpha=0.3)
    right_axis = left_axis.twinx()
    right_axis.plot(times, temperatures, color="steelblue", alpha=0.7, label="Reactor temperature")
    right_axis.set_ylabel("Reactor temperature (K)", color="steelblue")
    right_axis.tick_params(axis="y", labelcolor="steelblue")
    lines, labels = left_axis.get_legend_handles_labels()
    right_lines, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(lines + right_lines, labels + right_labels, loc="best")
    figure.tight_layout()

    return status, results, figure


demo = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(label="Time-ordered CSTR trajectory CSV", file_types=[".csv"], type="filepath"),
    outputs=[
        gr.Markdown(label="Latest prediction"),
        gr.Dataframe(label="Predictions", interactive=False),
        gr.Plot(label="Prediction trajectory"),
    ],
    title="CSTR Thermal Runaway Inference",
    description=(
        "Upload one time-ordered trajectory CSV with at least 15 rows and the columns: "
        "q, cAi, Ti, Tc, cA, T, and dT_dt. The app predicts the remaining time to the thermal cutoff."
    ),
    examples=[["example_trajectory.csv"]],
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
