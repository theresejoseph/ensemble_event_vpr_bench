from vpr_methods_evaluation import parse
from parser_config import get_parser, apply_defaults
from datasets.brisbane_events import BrisbaneEventDataset, Brisbane_RGB_Dataset
import importlib
import numpy as np
from datasets.nsvap import NSVAPDataset, NSAVP_RGB_Dataset
import argparse
import pandas as pd
import os
from tqdm import tqdm
import itertools
import json
import csv
from pathlib import Path
import shutil

def default_args_training():
    parser_train = argparse.ArgumentParser()

    parser_train.add_argument("--method", type=str, default="mixvpr", help="vpr method to use")
    parser_train.add_argument("--dataset_type", type=str, default="Brisbane", help="dataset type (e.g., Brisbane, NSAVP)")
    
    parser_train.add_argument("--reconstruct_method_name", type=str, default="eventCount", help="Reconstruction method name (e.g., eventCount, timeSurface, e2vid)")
    # parser_train.add_argument("--ref_seq_idx", type=int, default=3, help="Reference sequence index")
    # parser_train.add_argument("--qry_seq_idx", type=int, default=0, help="Query sequence index")
    parser_train.add_argument("--seq_len", type=int, default=1, help="Sequence length for VPR")
    parser_train.add_argument("--patch_or_frame", type=str, default="patch", help="Use 'patch' for patch-based, 'frame' for frame-based reconstruction")
    parser_train.add_argument("--patch_num_cols", type=int, default="3", help="Num patch columns for patch-based reconstruction")
    parser_train.add_argument("--patch_num_rows", type=int, default="2", help="Num patch rows for patch-based reconstruction")
    parser_train.add_argument('--grid_or_nest', type=str, default='grid', help='Use "grid" for grid-based patches or "nest" for nested patches')
    parser_train.add_argument('--nest_scale_factor', type=float, default=0.7, help='Scale factor for nested patches')
    parser_train.add_argument('--num_patches', type=int, default=4, help='Number of patches to use in the nest')

    # binning setup
    parser_train.add_argument("--bin_tag", type=str, default="manual_adaptive", help="Tag for adaptive bin naming")
    parser_train.add_argument("--adaptive_bin", type=int, default=0, help="Set 1 for adaptive binning, 0 otherwise")
    parser_train.add_argument("--time_res", type=float, default=1.0, help="Time resolution in seconds")
    parser_train.add_argument("--count_bin", type=int, default=0, help="Use event count binning if 1")
    parser_train.add_argument("--events_per_bin", type=int, default=100_000, help="Number of events per bin for count binning")
    

    # Manual tuning from Optuna (accept comma-separated lists)
    parser_train.add_argument("--max_bins", type=float, default=20, help="Maximum number of bins")
    parser_train.add_argument("--odom_weights", type=str, default="0.5422808353118097,0.038401205353768675,0,0", help="Comma-separated odometry weights")
    parser_train.add_argument("--max_odoms", type=str, default="4.595340126749333,16.344098561823436,1,10", help="Comma-separated odometry max thresholds")
    parser_train.add_argument("--use_exponential", type=int, default=0, help="Use exponential binning if 1, linear otherwise")
    
    parser_train.add_argument("--run_idxs", type=int, nargs='+', default=None, help="Indices of combinations to run from the combo list")
    args_cli = parser_train.parse_args()

    args_cli.sequences=["night","morning", "sunrise", "sunset1", "sunset2", "daytime",
     'R0_FA0', 'R0_FS0', 'R0_FN0', 'R0_RA0', 'R0_RS0', 'R0_RN0', "night_training",
     "morning_training", "sunrise_training", "sunset1_training", "sunset2_training", "daytime_training",]
    args_cli.experiment_pairs = [ (15,12),(15, 13), (15, 14), (15, 16), (15, 17)] # Example pairs for training
    
    return args_cli



def args_for_load_save(args_cli, reconstruct_method_name):
    parser = get_parser()
    args = parser.parse_args([])
    args.dataset_type = args_cli.dataset_type
    args = apply_defaults(args)
    args.reconstruct_method_name = reconstruct_method_name

    # Override parsed args with specific values
    args.adaptive_bin = args_cli.adaptive_bin
    args.max_odoms = args_cli.max_odoms
    args.odom_weights = args_cli.odom_weights
    args.max_bins = args_cli.max_bins
    args.save_frames_video = 0
    args.sequences = args_cli.sequences
    args.adaptive_bin_tag = args_cli.bin_tag
    args.use_exponential = args_cli.use_exponential  # Use exponential binning if True, linear otherwise
    args.time_res = args_cli.time_res
    args.count_bin = args_cli.count_bin  # 1 for event count binning, 0 for time binning
    args.events_per_bin = args_cli.events_per_bin  # Number of events per bin for eventCount reconstruction

    # Initialize dataset and reconstructor based on the parameters.
    if args.dataset_type.lower() == 'nsavp':
        dataset = NSVAPDataset(args.dataset_path)
    elif args.dataset_type.lower() == 'brisbane':
        dataset = BrisbaneEventDataset(args.dataset_path)
    else:
        raise ValueError("Unsupported dataset type")

    # Dynamically construct the module path
    if args.reconstruct_method_name!= 'RGB_camera':
        module_path = f"reconstruction.{args.reconstruct_method_name}"
        reconstruction_module = importlib.import_module(module_path)
        reconstructor_class = getattr(reconstruction_module, "EventReconstructor")
        reconstructor = reconstructor_class() 
    elif args.reconstruct_method_name == 'RGB_camera' and args.dataset_type.lower() == 'brisbane':
        reconstructor=None
        dataset = Brisbane_RGB_Dataset(args.dataset_path)

    return args, dataset, reconstructor



def args_for_vpr(args_cli, reconstruct_method_name, idR, idQ):
    args_vpr = parse.parse_arguments(args_cli.method)
    # Override parsed args
    args_vpr.idR = idR
    args_vpr.idQ = idQ
    args_vpr.sequences = args_cli.sequences
    args_vpr.dataset_type = args_cli.dataset_type
    args_vpr.reconstruct_method_name = reconstruct_method_name
    args_vpr.saveSimMat = False #<---------------------------------------------------- change this to True if you want to save the similarity matrix
    
    args_vpr.adaptive_bin = args_cli.adaptive_bin
    args_vpr.expTag = ''
    args_vpr.adaptive_bin_tag = args_cli.bin_tag 
    args_vpr.time_res = args_cli.time_res
    args_vpr.count_bin = args_cli.count_bin  # 1 for event count binning, 0 for time binning
    args_vpr.events_per_bin = args_cli.events_per_bin  # Number of events per bin for eventCount reconstruction
    args_vpr.patch_or_frame = args_cli.patch_or_frame
    args_vpr.patch_num_cols = args_cli.patch_num_cols  # Number of patch columns for patch-based reconstruction
    args_vpr.patch_num_rows = args_cli.patch_num_rows  # Number of patch
    args_vpr.grid_or_nest = args_cli.grid_or_nest  # Use 'grid' for grid-based patches or 'nest' for nested patches
    args_vpr.nest_scale_factor = args_cli.nest_scale_factor  # Scale factor for nested patches
    args_vpr.num_patches = args_cli.num_patches  # Number
    args_vpr.seq_len = args_cli.seq_len  # Sequence length for VPR
    
    return args_vpr



def process_pair(args_cli, reconstruct_method_name, idR, idQ):
    from load_and_save import load_save_data
    args_ls, dataset, reconstructor = args_for_load_save(args_cli, reconstruct_method_name)
    args_ls.ref_seq_idx = idR
    args_ls.qry_seq_idx = idQ

    frames_r, _ = load_save_data(dataset, reconstructor, args_ls, 'ref', return_data=False)
    frames_q, _ = load_save_data(dataset, reconstructor, args_ls, 'qry', return_data=False)

    return reconstruct_method_name, idR, idQ, [len(frames_r), len(frames_q)]



def run_experiment(args_cli):
    from vpr_methods_evaluation.main import run_vpr
    vpr_methods = ['megaloc', 'mixvpr', 'netvlad', 'cosplace']
    recon_methods = ['eventCount', 'timeSurface', 'e2vid', 'eventCount_noPolarity']
    exp_pairs = args_cli.experiment_pairs

    recalls, aucs = [], []

    combos = list(itertools.product(vpr_methods, recon_methods, exp_pairs))

    for vpr_method, recon_method, (idR, idQ) in tqdm(combos, desc="VPR Experiments"):
        args_cli.method = vpr_method
        args_cli.reconstruct_method_name = recon_method

        print(f"Running VPR for idR={idR}, idQ={idQ}, recon_method={recon_method}, method={vpr_method}")
        args_vpr = args_for_vpr(args_cli, recon_method, idR, idQ)
        recall_at_1, auc = run_vpr(args_vpr)

        if recall_at_1 is None:
            print("⚠ Skipping due to invalid input.")
            continue

        recalls.append(recall_at_1)
        aucs.append(auc)

    stats = {
        "mean_recall": np.mean(recalls) if recalls else np.nan,
        "std_recall": np.std(recalls) if recalls else np.nan,
        "min_recall": np.min(recalls) if recalls else np.nan,
        "max_recall": np.max(recalls) if recalls else np.nan,
        "median_recall": np.median(recalls) if recalls else np.nan,
        "mean_auc": np.mean(aucs) if aucs else np.nan,
        "std_auc": np.std(aucs) if aucs else np.nan,
        "min_auc": np.min(aucs) if aucs else np.nan,
        "max_auc": np.max(aucs) if aucs else np.nan,
        "median_auc": np.median(aucs) if aucs else np.nan,
        "num_pairs": len(recalls)
    }
    return stats



def run_patch_grid_search(args_cli): 
    SAVE_PATH = "./hpc/patch_grid_search_results.csv"
  
    if args_cli.run_idxs:
        # Load combo list and run the given indices
        with open("./hpc/combo_list.json", "r") as f:
            all_combos = json.load(f)

        save_path = "./hpc/patch_grid_search_results.csv"

        for i in args_cli.run_idxs:
            combo = all_combos[i]
            print(f"\n▶ Running combo {i}: {combo}")
            args_cli = default_args_training()
            for k, v in combo.items():
                setattr(args_cli, k, v)
            try:
                stats = run_experiment(args_cli)
                entry = {**combo, **stats}

                # Append to full results CSV
                if os.path.exists(save_path):
                    df = pd.read_csv(save_path)
                else:
                    df = pd.DataFrame(columns=[
                        "grid_or_nest", "patch_num_cols", "patch_num_rows",
                        "nest_scale_factor", "num_patches",
                        "mean_recall", "std_recall", "min_recall",
                        "max_recall", "median_recall",
                        "mean_auc", "std_auc", "min_auc", "max_auc", "median_auc",
                        "num_pairs"
                    ])
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
                df.to_csv(save_path, index=False)
                print(f"✅ Saved combo {i} to master CSV")

                # Save individual result
                individual_save_path = f"./hpc/results/combo_{args_cli.run_idxs[0]}-{args_cli.run_idxs[-1]}.csv"
                if os.path.exists(individual_save_path):
                    df_indiv = pd.read_csv(individual_save_path)
                else:
                    df_indiv = pd.DataFrame(columns=[
                        "grid_or_nest", "patch_num_cols", "patch_num_rows",
                        "nest_scale_factor", "num_patches",
                        "mean_recall", "std_recall", "min_recall",
                        "max_recall", "median_recall",
                        "mean_auc", "std_auc", "min_auc", "max_auc", "median_auc",
                        "num_pairs"
                    ])
                df_indiv = pd.concat([df_indiv, pd.DataFrame([entry])], ignore_index=True)
                df_indiv.to_csv(individual_save_path, index=False)
                print(f"✅ Saved combo {i} to master CSV")
                print(f"📝 Saved individual combo {i} to {individual_save_path}")

            except Exception as e:
                print(f"❌ Combo {i} failed: {e}")
    else:
        print("❌ Please provide --run_idxs to run patch combo batch.")



def save_displacement_per_traverse(args_cli):
    from load_and_save import make_paths
    from vpr_methods_evaluation.test_dataset import read_images_paths

    displacement_save_path = "./hpc/displacement_per_traverse1.csv"
    
    print(f"Saving displacement stats to: {displacement_save_path}")
    
    # Prepare CSV header
    with open(displacement_save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "traverse_name", "dataset", "binning_type", "binning_value",
            "num_frames", "mean_displacement", "std_displacement"
        ])
    
    print("Starting time-based binning evaluations...")
    for time_res in [0.1, 0.15, 0.2, 0.25, 0.5, 1.0]:
        print(f"\n  Processing time_res = {time_res}s")
        args_cli.time_res = time_res
        args_cli.adaptive_bin = 0
        args_cli.count_bin = 0
        for traverse_id in range(12):
            args_cli.dataset_type = 'NSAVP' if traverse_id >= 6 else 'Brisbane'
            traverse_name = args_cli.sequences[traverse_id]
            print(f"    Traverse {traverse_name} ({args_cli.dataset_type})...")
            make_paths(args_cli, traverse_name)
            database_folder = str(args_cli.save_images_dir)
            try:
                database_paths = read_images_paths(database_folder)
            except FileNotFoundError as e:
                print(f"[WARNING] Skipping missing folder: {database_folder}")
                continue
            if len(database_paths) < 2:
                print("      Skipping: not enough frames.")
                continue

            database_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in database_paths]
            ).astype(float)

            displacements = np.linalg.norm(np.diff(database_utms, axis=0), axis=1)
            mean_disp = np.mean(displacements)
            std_disp = np.std(displacements)

            print(f"      Frames: {len(database_paths)} | Mean disp: {mean_disp:.2f}m | Std disp: {std_disp:.2f}m")

            with open(displacement_save_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    traverse_name, args_cli.dataset_type, "time", time_res,
                    len(database_paths), mean_disp, std_disp
                ])
    
    print("\nStarting count-based binning evaluations...")
    for event_count in [100_000, 200_000, 300_000, 500_000, 1_000_000]:
        print(f"\n  Processing event_count = {event_count}")
        args_cli.events_per_bin = event_count
        args_cli.count_bin = 1
        for traverse_id in range(12):
            args_cli.dataset_type = 'NSAVP' if traverse_id >= 6 else 'Brisbane'
            traverse_name = args_cli.sequences[traverse_id]
            print(f"    Traverse {traverse_name} ({args_cli.dataset_type})...")
            make_paths(args_cli, traverse_name)
            database_folder = str(args_cli.save_images_dir)
            try:
                database_paths = read_images_paths(database_folder)
            except FileNotFoundError as e:
                print(f"[WARNING] Skipping missing folder: {database_folder}")
                continue
            if len(database_paths) < 2:
                print("      Skipping: not enough frames.")
                continue

            database_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in database_paths]
            ).astype(float)

            displacements = np.linalg.norm(np.diff(database_utms, axis=0), axis=1)
            mean_disp = np.mean(displacements)
            std_disp = np.std(displacements)

            print(f"      Frames: {len(database_paths)} | Mean disp: {mean_disp:.2f}m | Std disp: {std_disp:.2f}m")

            with open(displacement_save_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    traverse_name, args_cli.dataset_type, "count", event_count,
                    len(database_paths), mean_disp, std_disp
                ])
    
    print("\nDisplacement logging complete.")



def clean_csv():
    """
    Clean VPR CSVs:
    - Remove rows with empty AUC if patch_or_frame == 'frame'
    - Ensure patch_rows and patch_cols are set to 1 for all frame-based rows
    - Cast patch_rows/patch_cols to int
    - Remove duplicate entries
    """
    def clean_df(df, csv_path):
        # Ensure patch_rows and patch_cols exist
        if 'patch_rows' not in df.columns:
            df['patch_rows'] = pd.NA
        if 'patch_cols' not in df.columns:
            df['patch_cols'] = pd.NA

        # Fix patch_rows/cols for 'frame'
        frame_mask = df['patch_or_frame'] == 'frame'
        df.loc[frame_mask, 'patch_rows'] = 1
        df.loc[frame_mask, 'patch_cols'] = 1

        # Remove frame rows with missing AUC
        invalid_mask = frame_mask & (df['auc'].isna() | (df['auc'].astype(str).str.strip() == ""))
        removed_count = invalid_mask.sum()
        if removed_count > 0:
            print(f"🗑 Removing {removed_count} invalid 'frame' rows with missing AUC")
            df = df[~invalid_mask]

        # Convert patch_rows and patch_cols to integer (handle float/NaN safely)
        for col in ['patch_rows', 'patch_cols']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Remove duplicates ignoring only 'runtime'
        dedup_cols = [c for c in df.columns if c != 'runtime']
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep='first')
        after = len(df)
        if before > after:
            print(f"🧹 Removed {before - after} duplicate rows (ignoring 'runtime')")

        # Save
        df.to_csv(csv_path, index=False)
        print(f"✅ Cleaned and saved: {csv_path.name}")

    # Time-based
    time_res_list = [1.0, 0.5, 0.25, 0.2, 0.15, 0.1]
    for time_res in time_res_list:
        print(f"\n🔍 Cleaning for time_res: {time_res}")
        for dataset_type in ['Brisbane', 'NSAVP']:
            csv_path = Path(f"./results/vpr_results_{dataset_type}_fixed_timebins_{time_res}.csv")
            if not csv_path.exists():
                print(f"⚠ File not found: {csv_path}")
                continue
            print(f"📄 Processing {csv_path.name}")
            df = pd.read_csv(csv_path)
            clean_df(df, csv_path)

    # Count-based
    eventsPerBin_list = [100_000, 200_000, 300_000, 500_000, 1_000_000]
    for events_per_bin in eventsPerBin_list:
        print(f"\n🔍 Cleaning for countbin: {events_per_bin}")
        for dataset_type in ['Brisbane', 'NSAVP']:
            csv_path = Path(f"./results/vpr_results_{dataset_type}_fixed_countbins_{events_per_bin}.csv")
            if not csv_path.exists():
                print(f"⚠ File not found: {csv_path}")
                continue
            print(f"📄 Processing {csv_path.name}")
            df = pd.read_csv(csv_path)
            clean_df(df, csv_path)



def rename_simMat(args_cli):
    for patch_or_frame in ['patch', 'frame']:  # Options for patch or frame
        if patch_or_frame == 'patch':
            patch_rows, patch_cols = 2, 3
        else:
            patch_rows, patch_cols = 1, 1

        for dataset_type in ['Brisbane', 'NSAVP']:
            experiment_pairs = [(3, 0), (3, 1), (3, 2), (3, 4), (3, 5)] if dataset_type == 'Brisbane' else [(6, 7), (6, 8), (8, 7), (9, 11), (10, 11)]

            for method in ['mixvpr', 'megaloc', 'netvlad', 'cosplace']:
                for events_per_bin in [100_000, 200_000, 300_000, 500_000, 1_000_000]:
                    for reconstruct_method in ['e2vid', 'timeSurface', 'eventCount', 'eventCount_noPolarity']:
                        for idR, idQ in experiment_pairs:
                            ref_seq = args_cli.sequences[idR]
                            qry_seq = args_cli.sequences[idQ]

                            # Old path (without row/col suffix)
                            old_simMatPath = Path(f"./logs/{dataset_type}/fixed_countbins_{events_per_bin}/{ref_seq}_vs_{qry_seq}_{method}_l2_reconstruct_{reconstruct_method}_None_{patch_or_frame}.npy")
                            new_simMatPath = Path(f"./logs/{dataset_type}/fixed_countbins_{events_per_bin}/{ref_seq}_vs_{qry_seq}_{method}_l2_reconstruct_{reconstruct_method}_None_{patch_or_frame}_{patch_rows}_{patch_cols}.npy")

                            if old_simMatPath.exists():
                                if not new_simMatPath.exists():
                                    shutil.move(old_simMatPath, new_simMatPath)
                                    print(f"✅ Renamed: {old_simMatPath.name} → {new_simMatPath.name}")
                                else:
                                    print(f"⚠ Skipping rename: target already exists: {new_simMatPath.name}")
                            else:
                                print(f"❌ SimMat not found: {old_simMatPath}")
       




if __name__ == "__main__":
    args_cli = default_args_training()
    # run_patch_grid_search(args_cli)

    args, dataset, reconstructor = args_for_load_save(args_cli, args_cli.reconstruct_method_name)
    # save_displacement_per_traverse(args)

    # clean_csv()
    # rename_simMat(args_cli)


