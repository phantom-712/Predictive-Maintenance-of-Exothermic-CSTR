import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Configuration & Model Definition
# ==========================================
# Updated paths for Colab environment
CSV_FILE = "/content/cstr_runaway_trajectories.csv"
MODEL_PATH = "/content/cstr_lstm_model.pth"
WINDOW_SIZE = 15
FEATURE_COLS = ["q", "cAi", "Ti", "Tc", "cA", "T", "dT_dt"]
TARGET_COL = "time_to_cutoff"

# Redefine the exact architecture used in training
class CSTRRunawayLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSTRRunawayLSTM(input_dim=len(FEATURE_COLS)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully.")

# ==========================================
# 2. Data Loading & Recreating Scaler
# ==========================================
# We must scale the inference data exactly as we scaled the training data
df = pd.read_csv(CSV_FILE)
unique_runs = df["run_id"].unique()
train_runs = unique_runs[:80]
test_runs = unique_runs[80:]

df_train = df[df["run_id"].isin(train_runs)].copy()
df_test = df[df["run_id"].isin(test_runs)].copy()

scaler = StandardScaler()
scaler.fit(df_train[FEATURE_COLS])  # Fit strictly on train data

# ==========================================
# 3. Select a Run for Inference
# ==========================================
# Try to find a runaway in the test set
runaway_test_runs = df_test[df_test["is_runaway"] == 1]["run_id"].unique()
runaway_train_runs = df_train[df_train["is_runaway"] == 1]["run_id"].unique()

if len(runaway_test_runs) > 0:
    test_run_id = runaway_test_runs[0]
    print(f"Found runaway in Test Set. Running inference on Test Run ID: {test_run_id}")

elif len(runaway_train_runs) > 0:
    test_run_id = runaway_train_runs[0]
    print(f"No runaways in test set. Found in Train Set. Running inference on Train Run ID: {test_run_id}")

else:
    # No runaways exist in the entire CSV.
    # Find the run that got the HOTTEST so we have something dynamic to look at.
    print("WARNING: There are 0 runaway events (T >= 400K) in your entire dataset!")
    print("Selecting the run that reached the highest peak temperature instead...")

    # Get index of the max temperature in the test set
    max_t_idx = df_test["T"].idxmax()
    test_run_id = df_test.loc[max_t_idx, "run_id"]
    max_temp = df_test.loc[max_t_idx, "T"]
    print(f"Running inference on Test Run ID: {test_run_id} (Peak Temp: {max_temp:.2f} K)")

# Retrieve the dataframe for the chosen run (from the full df to be safe)
run_df = df[df["run_id"] == test_run_id].copy()

# Scale features
scaled_features = scaler.transform(run_df[FEATURE_COLS])
actual_rul = run_df[TARGET_COL].values
time_axis = run_df["time"].values
temperatures = run_df["T"].values

# ==========================================
# 4. Run Sliding Window Inference
# ==========================================
predictions = []
valid_times = []
valid_actuals = []
valid_temps = []

with torch.no_grad():
    for i in range(len(scaled_features) - WINDOW_SIZE + 1):
        # Extract the window
        window = scaled_features[i : i + WINDOW_SIZE]
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        pred = model(window_tensor)
        predictions.append(pred.item())

        # Record the corresponding ground truth (end of the window)
        target_idx = i + WINDOW_SIZE - 1
        valid_times.append(time_axis[target_idx])
        valid_actuals.append(actual_rul[target_idx])
        valid_temps.append(temperatures[target_idx])

# ==========================================
# 5. Plot Results
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Actual vs Predicted RUL (Time to Cutoff)
ax1.plot(valid_times, valid_actuals, 'g--', label="Actual Time-to-Cutoff", linewidth=2)
ax1.plot(valid_times, predictions, 'b-', label="Predicted Time-to-Cutoff", linewidth=2)
ax1.set_xlabel("Simulation Time (min)")
ax1.set_ylabel("Time to Runaway (min)", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(0, 21)

# Plot Reactor Temperature on secondary axis for context
ax2 = ax1.twinx()
ax2.plot(valid_times, valid_temps, 'r-', alpha=0.5, label="Reactor Temp (K)", linewidth=2)
ax2.axhline(y=400, color='r', linestyle=':', label="Critical Cutoff (400 K)")
ax2.set_ylabel("Reactor Temperature (K)", color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title(f"LSTM Inference Check: CSTR Thermal Runaway (Run {test_run_id})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()