import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_safety_and_prognostic_metrics(
    df_results,
    true_col='y_true',
    pred_col='y_pred',
    run_id_col='run_id',
    time_col='time',
    is_runaway_col='is_runaway',
    crit_threshold=5.0,
    action_threshold=3.0,
    late_epsilon=0.5,
):
    """Evaluates prognostic performance on CSTR Time-to-Cutoff predictions."""
    y_true = df_results[true_col].values
    y_pred = df_results[pred_col].values
    error = y_pred - y_true  

    # 1. NASA / PHM Asymmetric Prognostic Scoring Function
    a1, a2 = 10.0, 5.0
    phm_penalties = np.where(error < 0, np.exp(-error / a1) - 1.0, np.exp(error / a2) - 1.0)
    phm_score = float(np.sum(phm_penalties))
    norm_phm_score = float(np.mean(phm_penalties))

    # 2. General Regression Metrics (Global)
    rmse_global = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae_global = float(np.mean(np.abs(y_pred - y_true)))

    # 3. Zone-Specific Metrics (Critical Zone: y_true <= crit_threshold)
    crit_mask = (y_true <= crit_threshold) & (df_results[is_runaway_col] == 1)
    if np.sum(crit_mask) > 0:
        y_true_crit = y_true[crit_mask]
        y_pred_crit = y_pred[crit_mask]
        err_crit = y_pred_crit - y_true_crit

        rmse_crit = float(np.sqrt(np.mean(err_crit**2)))
        mae_crit = float(np.mean(np.abs(err_crit)))
        late_preds = np.sum(err_crit > late_epsilon)
        late_prediction_rate = float((late_preds / len(y_true_crit)) * 100.0)
    else:
        rmse_crit, mae_crit, late_prediction_rate = 0.0, 0.0, 0.0

    # 4. Operational Trip Metrics (Lead Time & False Alarms)
    lead_times = []
    missed_detections = 0
    false_alarms = 0

    for run_id, group in df_results.groupby(run_id_col):
        is_runaway = group[is_runaway_col].iloc[0] == 1
        alarm_steps = group[group[pred_col] <= action_threshold]

        if is_runaway:
            t_cutoff = group[group[true_col] == 0.0]
            t_breach = t_cutoff[time_col].iloc[0] if len(t_cutoff) > 0 else group[time_col].max()
            if len(alarm_steps) > 0:
                t_first_alarm = alarm_steps[time_col].iloc[0]
                lead_time = max(0.0, t_breach - t_first_alarm)
                lead_times.append(lead_time)
            else:
                missed_detections += 1
                lead_times.append(0.0)
        else:
            if len(alarm_steps) > 0:
                false_alarms += 1

    mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
    total_stable_runs = (df_results.groupby(run_id_col)[is_runaway_col].max() == 0).sum()
    false_alarm_rate = float((false_alarms / total_stable_runs) * 100.0) if total_stable_runs > 0 else 0.0

    return {
        'PHM Score (Total)': phm_score,
        'PHM Score (Normalized / Step)': norm_phm_score,
        'Global RMSE (min)': rmse_global,
        'Global MAE (min)': mae_global,
        'Critical Zone RMSE (min)': rmse_crit,
        'Critical Zone MAE (min)': mae_crit,
        'Late Prediction Rate (%)': late_prediction_rate,
        'Mean Lead Time (min)': mean_lead_time,
        'Missed Runaways (Count)': missed_detections,
        'False Alarm Rate (%)': false_alarm_rate,
    }


def plot_trajectory_evaluation(df_run, run_id, time_col='time', true_col='y_true', pred_col='y_pred'):
    """Plots ground truth vs predicted time-to-cutoff for a single trajectory."""
    plt.figure(figsize=(9, 4.5))
    plt.plot(df_run[time_col], df_run[true_col], 'k--', label='Ground Truth Time-to-Cutoff', linewidth=2)
    plt.plot(df_run[time_col], df_run[pred_col], 'r-', label='LSTM Predicted Time-to-Cutoff', linewidth=2)

    plt.axhline(y=5.0, color='orange', linestyle=':', label='Critical Warning Zone (5 min)')
    plt.axhline(y=0.0, color='gray', linestyle='-', alpha=0.7, label='Cutoff Breach (0 min)')

    plt.title(f'CSTR Runaway Emergency Advisory — Run {run_id}', fontsize=12, fontweight='bold')
    plt.xlabel('Simulation Time (minutes)')
    plt.ylabel('Remaining Safe Time (minutes)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save to disk in case the GUI window hangs in VS Code
    plt.savefig(f"evaluation_run_{run_id}.png")
    print(f"Plot saved to evaluation_run_{run_id}.png", flush=True)
    plt.show()


if __name__ == "__main__":
    print("Starting evaluation script...", flush=True)
    
    csv_file = "cstr_runaway_trajectories.csv"
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Ensure it is in the same directory.", flush=True)
        exit()
        
    print(f"Loading data from {csv_file}...", flush=True)
    df = pd.read_csv(csv_file)
    
    # Generate mock LSTM predictions if the column doesn't exist yet
    if 'y_pred' not in df.columns:
        print("No 'y_pred' column found from a model. Generating simulated predictions for testing...", flush=True)
        # Add slight random noise leaning conservative (early)
        np.random.seed(42)
        noise = np.random.normal(loc=-0.2, scale=0.8, size=len(df))
        df['y_pred'] = np.clip(df['time_to_cutoff'] + noise, 0, 20.0)
    
    print("Computing evaluation metrics...", flush=True)
    metrics = compute_safety_and_prognostic_metrics(
        df, 
        true_col='time_to_cutoff', 
        pred_col='y_pred',
        run_id_col='run_id',
        time_col='time',
        is_runaway_col='is_runaway'
    )
    
    print("\n" + "=" * 45, flush=True)
    print("     CSTR SAFETY & PROGNOSTIC REPORT", flush=True)
    print("=" * 45, flush=True)
    for k, v in metrics.items():
        print(f"{k:<32}: {v:.3f}", flush=True)
    print("=" * 45 + "\n", flush=True)
    
    runaway_runs = df[df['is_runaway'] == 1]['run_id'].unique()
    if len(runaway_runs) > 0:
        sample_run_id = runaway_runs[0]
        print(f"Plotting evaluation for Run {sample_run_id}...", flush=True)
        sample_run = df[df['run_id'] == sample_run_id]
        plot_trajectory_evaluation(
            sample_run, 
            run_id=sample_run_id, 
            true_col='time_to_cutoff', 
            pred_col='y_pred'
        )
    else:
        print("No runaway events found in the dataset to plot.", flush=True)