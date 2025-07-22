import os
import argparse
import time 
import numpy as np
import optuna
from training import default_args_training, run_experiment
from load_and_save import load_save_data, make_paths
parser = argparse.ArgumentParser(description="Hyperparameter Optimisation with Optuna")
parser.add_argument("--expName", type=str, default='debug', help="Experiment Number")
args = parser.parse_args()
import csv

CSV_PATH = "/mnt/hpccs01/home/n10234764/event_vo_vpr/hpc/patch_matching_params_nest+grid.csv"
CSV_HEADER = ["trial_number", "grid_or_nest", "patch_num_rows", "patch_num_cols", "num_patches", "nest_scale_factor",
              "min_recall", "median_recall", "max_recall", "std_recall", "avg_recall", "score"]

def append_trial_to_csv(trial, patch_num_rows, patch_num_cols, nest_scale, num_patches,min_recall, median_recall, max_recall, std_recall, avg_recall, score):
    is_new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(CSV_HEADER)
        writer.writerow([
            trial.number,
            trial.params["grid_or_nest"],
            num_patches,
            patch_num_rows,
            patch_num_cols,
            nest_scale,
            round(min_recall, 2),
            round(median_recall, 2),
            round(max_recall, 2),
            round(std_recall, 2),
            round(avg_recall, 4),
            round(score, 4)
        ])


def objective(trial):
    # --- Load default args ---
    vpr_args = default_args_training()

    # --- Suggest hyperparameters ---
    grid_or_nest = trial.suggest_categorical("grid_or_nest", ["grid", "nest"])
    vpr_args.grid_or_nest = grid_or_nest # Use grid-based patches

    patch_cols, patch_rows, nest_scale_factor, num_patches = None, None, None, None
    if grid_or_nest == 'grid':
        patch_rows = trial.suggest_int("patch_num_rows", 1, 10)
        patch_cols = trial.suggest_int("patch_num_cols", 1, 10)
        num_patches = patch_rows * patch_cols

        vpr_args.patch_num_rows = patch_rows
        vpr_args.patch_num_cols = patch_cols
        vpr_args.num_patches = num_patches  # Number of patches to use in

    else:  # nested patches
        nest_scale_factor = trial.suggest_float("nest_scale_factor", 0.5, 0.99)
        num_patches = trial.suggest_int("num_patches", 1, 10)
    
        vpr_args.nest_scale_factor = nest_scale_factor  # Scale factor for nested patches
        vpr_args.num_patches = num_patches  # Number of patches to use in



    # --- Run VPR method ---
    recalls = run_experiment(vpr_args)

    # --- Aggregate recall stats ---
    min_recall = np.min(recalls)
    median_recall = np.median(recalls)
    max_recall = np.max(recalls)
    std_recall = np.std(recalls)
    avg_recall = np.mean(recalls)

    # --- Penalize patch count ---
    score = avg_recall

    # --- Log to CSV ---
    append_trial_to_csv(trial, patch_rows, patch_cols, nest_scale_factor, num_patches, min_recall, median_recall, max_recall, std_recall, avg_recall, score)

    print(f"Trial {trial.number}: {grid_or_nest}, scale_factor={nest_scale_factor}, rows={patch_rows}, cols={patch_cols}, "
          f"patches={num_patches}, score={score:.4f}")

    return score

if __name__ == "__main__":
    db_path = f"/mnt/hpccs01/home/n10234764/event_vo_vpr/hpc/patch_matching_param_optim2.db"
    study_name = "patch_matching_param_optim"
    try:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        optuna.logging.set_verbosity(optuna.logging.INFO)

    except KeyError:
        study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{db_path}", direction="maximize")

    study.optimize(objective, n_trials=500)

    print("Best trial:")
    print(study.best_trial)
