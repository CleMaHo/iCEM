# logging_utils.py

import csv
import json
import os
import datetime
import numpy as np

def log_results_and_parameters(
    trial_results,            # List of (converged: bool, steps_to_convergence: int)
    MAX_ITERATIONS,
    horizon,
    dt,
    control_dim,
    NUM_SAMPLES,
    TEMPERATURE,
    STD_DEV,
    Q_val,
    R_val,
    S_val,
    cutoff,
    fs,
    filter_order,
    csv_filename="trial_summary_log.csv",
    json_dir="logs_json"
):
    # Aggregate metrics
    trials = len(trial_results)
    converged_trials = [steps for converged, steps in trial_results if converged]
    num_converged = len(converged_trials)
    num_non_converged = trials - num_converged
    success_rate = num_converged / trials if trials > 0 else 0.0

    mean_steps = float(np.mean(converged_trials)) if converged_trials else None
    std_steps = float(np.std(converged_trials)) if converged_trials else None
    min_steps = float(np.min(converged_trials)) if converged_trials else None
    max_steps = float(np.max(converged_trials)) if converged_trials else None
    timestamp = datetime.datetime.now().isoformat()
    
    # ===== Write to CSV =====
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                "datetime", "trials", "converged", "non_converged", "success_rate",
                "max_iterations", "mean_steps_to_converge", "std_steps_to_converge",
                "max_steps_to_converge", "min_steps_to_converge",
                "horizon", "dt", "control_dim", "NUM_SAMPLES", "TEMPERATURE",
                "STD_DEV", "Q", "R", "S", "cutoff", "fs", "filter_order"
            ])
        writer.writerow([
            timestamp, trials, num_converged, num_non_converged, round(success_rate, 3), MAX_ITERATIONS,
            round(mean_steps, 3) if mean_steps is not None else "",
            round(std_steps, 3) if std_steps is not None else "",
            max_steps if max_steps is not None else "",
            min_steps if min_steps is not None else "",
            horizon, dt, control_dim, NUM_SAMPLES, 
            TEMPERATURE, STD_DEV, Q_val, R_val, S_val,
            cutoff, fs, filter_order
        ])

    # ===== Write to JSON =====
    os.makedirs(json_dir, exist_ok=True)
    json_data = {
        "datetime": timestamp,
        "parameters": {
            "max_iterations": MAX_ITERATIONS,
            "horizon": horizon,
            "dt": dt,
            "control_dim": control_dim,
            "NUM_SAMPLES": NUM_SAMPLES,
            "TEMPERATURE": TEMPERATURE,
            "STD_DEV": STD_DEV,
            "Q": Q_val,
            "R": R_val,
            "S": S_val,
            "cutoff": cutoff,
            "fs": fs,
            "filter_order": filter_order
        },
        "trial_results": [
            {"trial": i + 1, "converged": bool(converged), "steps": steps}
            for i, (converged, steps) in enumerate(trial_results)
        ],
        "summary": {
            "total_trials": trials,
            "converged": num_converged,
            "non_converged": num_non_converged,
            "success_rate": round(success_rate, 3),
            "max_iterations": MAX_ITERATIONS,
            "mean_steps_to_converge": mean_steps,
            "std_steps_to_converge": std_steps,
            "max_steps_to_converge": max_steps,
            "min_steps_to_converge": min_steps,
        }
    }

    json_filename = os.path.join(json_dir, f"log_{timestamp.replace(':', '-')}.json")
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=2)
