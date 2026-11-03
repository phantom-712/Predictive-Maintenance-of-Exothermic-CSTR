"""Train the LSTM used by task_2.py on generated CSTR trajectories."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "cstr_runaway_trajectories.csv"
MODEL_PATH = SCRIPT_DIR / "cstr_lstm_model.pth"
SCALER_PATH = SCRIPT_DIR / "cstr_feature_scaler.joblib"
WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
FEATURE_COLS = ["q", "cAi", "Ti", "Tc", "cA", "T", "dT_dt"]
TARGET_COL = "time_to_cutoff"


class CSTRSequenceDataset(Dataset):
    """Fixed-length sequences that never span two reactor simulations."""

    def __init__(self, data_df, window_size=WINDOW_SIZE):
        sequences, targets = [], []
        for _, group in data_df.groupby("run_id"):
            features = group[FEATURE_COLS].values
            rul = group[TARGET_COL].values
            for index in range(len(group) - window_size + 1):
                sequences.append(features[index : index + window_size])
                targets.append(rul[index + window_size - 1])

        self.x = torch.tensor(np.asarray(sequences), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(targets), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class AsymmetricRULLoss(nn.Module):
    """Apply a stronger penalty when remaining-safe-time is overestimated."""

    def __init__(self, late_penalty=2.5):
        super().__init__()
        self.late_penalty = late_penalty

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        loss = torch.where(error > 0, self.late_penalty * error.square(), error.square())
        return loss.mean()


class CSTRRunawayLSTM(nn.Module):
    """Architecture shared exactly with task_2.py."""

    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def average_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            total_loss += criterion(model(x_batch), y_batch).item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Trajectory data not found: {CSV_FILE}. Run task_1.py first.")

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from {CSV_FILE.name}...")
    dataframe = pd.read_csv(CSV_FILE)

    unique_runs = dataframe["run_id"].unique()
    train_runs, test_runs = unique_runs[:80], unique_runs[80:]
    train_frame = dataframe[dataframe["run_id"].isin(train_runs)].copy()
    test_frame = dataframe[dataframe["run_id"].isin(test_runs)].copy()

    scaler = StandardScaler()
    train_frame[FEATURE_COLS] = scaler.fit_transform(train_frame[FEATURE_COLS])
    test_frame[FEATURE_COLS] = scaler.transform(test_frame[FEATURE_COLS])

    train_dataset = CSTRSequenceDataset(train_frame)
    test_dataset = CSTRSequenceDataset(test_frame)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train sequences: {len(train_dataset)} | Test sequences: {len(test_dataset)}")

    model = CSTRRunawayLSTM(input_dim=len(FEATURE_COLS)).to(device)
    criterion = AsymmetricRULLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x_batch.size(0)

        if epoch == 1 or epoch % 5 == 0:
            train_loss = total_train_loss / len(train_loader.dataset)
            test_loss = average_loss(model, test_loader, criterion, device)
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
            )

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}")
    print(f"Feature scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    main()
