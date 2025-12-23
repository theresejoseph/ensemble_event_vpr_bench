import os
import argparse
import time 
import numpy as np
import optuna
from training import args_for_vpr, args_for_load_save
from load_and_save import load_save_data, make_paths
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from vpr_methods_evaluation.main import run_vpr
parser = argparse.ArgumentParser(description="Hyperparameter Optimisation with Optuna")
parser.add_argument("--expName", type=str, default='debug', help="Experiment Number")
args = parser.parse_args()
import csv

CSV_PATH = "/mnt/hpccs01/home/n10234764/event_vo_vpr/hpc/adaptive_binning5_trials.csv"
CSV_HEADER = ["trial_number", "max_bins", "use_exponential",
              "w_ang_vel", "w_lin_speed", "w_ang_acc", "w_lin_acc",
              "max_ang_vel", "max_lin_speed", "max_ang_acc", "max_lin_acc",
              "avg_frames", "avg_recall", "score"]


def append_trial_to_csv(trial, avg_frames, avg_recall, score):
    is_new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(CSV_HEADER)
        writer.writerow([
            trial.number,
            trial.params["max_bins"],
            trial.params["use_exponential"],
            trial.params["w_ang_vel"],
            trial.params["w_lin_speed"],
            trial.params["w_ang_acc"],
            trial.params["w_lin_acc"],
            trial.params["max_ang_vel"],
            trial.params["max_lin_speed"],
            trial.params["max_ang_acc"],
            trial.params["max_lin_acc"],
            round(avg_frames, 2),
            round(avg_recall, 4),
            round(score, 4)
        ])

def process_pair(rep, i, max_odoms, weights, max_bins, use_exponential):
    args_ls, dataset, reconstructor = args_for_load_save(rep, max_odoms, weights, max_bins, use_exponential)
    args_ls.ref_seq_idx = i
    args_ls.qry_seq_idx = i+1 if i+1 < len(args_ls.sequences) else i

    frames_r, _ = load_save_data(dataset, reconstructor, args_ls, 'ref')
    frames_q, _ = load_save_data(dataset, reconstructor, args_ls, 'qry')

    return rep, i, [len(frames_r), len(frames_q)]



def run_vpr_wrapper(idR, idQ, rep, max_bins, subfolder_dir):
    args_vpr = args_for_vpr(idR, idQ, rep, max_bins)
    args_vpr.subfolder_dir = subfolder_dir
    args_vpr.saveSimMat = False
    return run_vpr(args_vpr)




def objective(trial):
    # Trial parameters
    max_bins = trial.suggest_int("max_bins", 10, 50)
    use_exponential = trial.suggest_categorical("use_exponential", [True, False])

    # Weight distribution (optional: normalize if needed)
    w_ang_vel = trial.suggest_float("w_ang_vel", 0.0, 1.0)
    w_lin_speed = trial.suggest_float("w_lin_speed", 0.0, 1.0)
    w_ang_acc = trial.suggest_float("w_ang_acc", 0.0, 1.0)
    w_lin_acc = trial.suggest_float("w_lin_acc", 0.0, 1.0)
    weights = [w_ang_vel, w_lin_speed, w_ang_acc, w_lin_acc] #w_ang_vel, w_lin_speed, w_ang_acc, w_lin_acc 

    # Trial parameters for motion limits
    max_ang_vel = trial.suggest_float("max_ang_vel", 0.5, 5.0)
    max_lin_speed = trial.suggest_float("max_lin_speed", 10.0, 30.0)
    max_ang_acc = trial.suggest_float("max_ang_acc", 0.5, 5.0)
    max_lin_acc = trial.suggest_float("max_lin_acc", 10.0, 30.0)
    max_odoms = [max_ang_vel, max_lin_speed, max_ang_acc, max_lin_acc]

    # Clean up: delete the subfolder directory if it exists after every run
    args_load_save, dataset, reconstructor = args_for_load_save( 'eventCount', max_odoms, weights, max_bins, use_exponential)
    make_paths(args_load_save, 'night_training')
    if hasattr(args_load_save, 'subfolder_dir') and os.path.exists(args_load_save.subfolder_dir):
        shutil.rmtree(args_load_save.subfolder_dir)
        print(f"Deleted directory: {args_load_save.subfolder_dir}")

    # Load and save
    # Early pruning check: run synchronously to avoid wasting compute
    rep = 'eventCount'
    i = 0
    args_ls, dataset, reconstructor = args_for_load_save(rep, max_odoms, weights, max_bins, use_exponential)
    args_ls.ref_seq_idx = i
    args_ls.qry_seq_idx = i+1 if i+1 < len(args_ls.sequences) else i

    frames_r, _ = load_save_data(dataset, reconstructor, args_ls, 'ref')
    if len(frames_r) > 1000:
        print(f"Early prune: eventCount @ i=0 too many frames ({len(frames_r)})")
        raise optuna.exceptions.TrialPruned()
    frames_q, _ = load_save_data(dataset, reconstructor, args_ls, 'qry')
    all_num_frames = [len(frames_r), len(frames_q)]

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        for rep in ['eventCount', 'timeSurface', 'e2vid']:
            for i in range(0, len(args_ls.sequences), 2):
                if rep == 'eventCount' and i == 0:
                    continue  # Already processed above
                futures.append(executor.submit(process_pair, rep, i, max_odoms, weights, max_bins, use_exponential))

        for future in as_completed(futures):
            _, _, lengths = future.result()
            all_num_frames.extend(lengths)
    avg_frames = np.mean(all_num_frames)

    # Run VPR
    recall_jobs = []
    subfolder_dir = os.path.basename(args_load_save.subfolder_dir)
    with ProcessPoolExecutor(max_workers=12) as executor:
        for rep in ['eventCount', 'timeSurface', 'e2vid']:
            for idR, idQ in [(3, 0), (3, 1), (3, 2), (3, 4), (3, 5)]:
                recall_jobs.append(executor.submit(run_vpr_wrapper, idR, idQ, rep, max_bins, subfolder_dir))

        recalls = [future.result() for future in recall_jobs]



    avg_recall = np.mean(recalls)

    # Objective: Maximize recall, penalize large number of frames
    objective_score = avg_recall
    print(f"Recall: {avg_recall:.4f}, Avg Frames: {avg_frames:.1f}, Score: {objective_score:.4f}")

    # Log to CSV
    append_trial_to_csv(trial, avg_frames, avg_recall, objective_score)

    return objective_score




def main():
    db_path = f"/mnt/hpccs01/home/n10234764/event_vo_vpr/hpc/_adaptive_binning5.db"
    study_name = "adaptive_binning_optim"

    try:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        optuna.logging.set_verbosity(optuna.logging.INFO)

    except KeyError:
        study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{db_path}", direction="maximize")

    study.optimize(objective, n_trials=500)

    print("Best trial:")
    print(study.best_trial)


if __name__ == "__main__":
    main()
