"""Export the trained CSTR predictor for the free browser-only Hugging Face Space."""

import json
from pathlib import Path

import joblib
import onnx
import pandas as pd
import torch

from train_model import CSTRRunawayLSTM, FEATURE_COLS, MODEL_PATH, SCALER_PATH, WINDOW_SIZE


SPACE_DIR = Path(__file__).resolve().parents[1] / "huggingface_static_space"
ONNX_PATH = SPACE_DIR / "model.onnx"
SCALER_JSON_PATH = SPACE_DIR / "scaler.json"
PROFILES_JSON_PATH = SPACE_DIR / "default_profiles.json"
TRAJECTORIES_PATH = Path(__file__).resolve().parent / "cstr_runaway_trajectories.csv"


def main():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing. Run train_model.py first.")

    SPACE_DIR.mkdir(exist_ok=True)
    model = CSTRRunawayLSTM(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    example_input = torch.zeros(1, WINDOW_SIZE, len(FEATURE_COLS), dtype=torch.float32)
    torch.onnx.export(
        model,
        example_input,
        ONNX_PATH,
        input_names=["features"],
        output_names=["remaining_safe_time"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(ONNX_PATH))

    scaler = joblib.load(SCALER_PATH)
    SCALER_JSON_PATH.write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLS,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "window_size": WINDOW_SIZE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    trajectories = pd.read_csv(TRAJECTORIES_PATH)
    profiles = []
    for profile_id, run_id, title, description, default_scenario in [
        ("stable", 1, "Safe control", "The Task 1 safe-control branch: no upset is injected.", "safe"),
        (
            "runaway",
            10,
            "Feed-surge upset",
            "Task 1 feed-surge logic: 60% higher feed concentration and 70% lower cooling after the sampled trigger time.",
            "feed_surge",
        ),
    ]:
        source_run = trajectories[trajectories["run_id"] == run_id].copy()
        run = source_run
        if profile_id == "stable":
            run = run.iloc[-45:]
        profiles.append(
            {
                "id": profile_id,
                "title": title,
                "description": description,
                "default_scenario": default_scenario,
                "initial_state": {"cA": float(source_run["cA"].iloc[0]), "T": float(source_run["T"].iloc[0])},
                "base_inputs": {column: float(source_run[column].iloc[0]) for column in FEATURE_COLS[:4]},
                "defaults": {column: float(source_run[column].iloc[0]) for column in FEATURE_COLS[:4]},
                "records": run[["time", *FEATURE_COLS]].to_dict(orient="records"),
            }
        )
    PROFILES_JSON_PATH.write_text(json.dumps(profiles, separators=(",", ":")), encoding="utf-8")
    print(f"Exported ONNX model: {ONNX_PATH}")
    print(f"Exported scaler configuration: {SCALER_JSON_PATH}")
    print(f"Exported {len(profiles)} interactive default profiles: {PROFILES_JSON_PATH}")


if __name__ == "__main__":
    main()
