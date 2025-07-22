
import numpy as np
from pathlib import Path
from vpr_methods_evaluation.test_dataset import TestDataset
from scipy.ndimage import uniform_filter1d
import itertools
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--ref_seq', required=True)
    parser.add_argument('--qry_seq', required=True)
    parser.add_argument('--ensemble_over', required=True)
    parser.add_argument('--vpr_methods', required=True)
    parser.add_argument('--recon_methods', required=True)
    parser.add_argument('--time_strs', required=True)
    parser.add_argument('--seq_len', type=int, default=1)
    return parser.parse_args()


def apply_sequence_matching(sim_matrix, seq_len, seq_match_type='modified'):
    """
    Applies sequence matching row-wise on the entire similarity matrix.
    Returns a new similarity matrix after applying the seq_match_row logic.
    """
    Q, R = sim_matrix.shape
    matched_sim_matrix = np.zeros_like(sim_matrix)

    # Patch to allow testing both modes without global args
    class Args:
        pass
    args = Args()
    args.seq_match_type = seq_match_type

    for i in range(Q):
        if seq_len == 1:
            matched_sim_matrix[i] = sim_matrix[i]   
        else:   
            matched_sim_matrix[i] = seq_match_row_standard(sim_matrix, i, seq_len, seq_match_type=seq_match_type)

    return matched_sim_matrix



def seq_match_row_standard(simMat, row_idx, seq_len, seq_match_type='modified'):
    Q, R = simMat.shape

    if seq_match_type == 'seqslam':
        if row_idx < seq_len - 1:
            return np.full(R, -np.inf)
        
        local_mean = uniform_filter1d(simMat, size=min(20, R//4), axis=1, mode='reflect')
        local_std = np.sqrt(uniform_filter1d((simMat - local_mean)**2, size=min(20, R//4), axis=1, mode='reflect')) + 1e-8
        enhanced_sim = (simMat - local_mean) / local_std
        
        seq_scores = np.full(R, -np.inf)
        min_velocity = 0.8
        max_velocity = 1.2
        
        for j in range(seq_len - 1, R):
            best_score = -np.inf
            for v in np.linspace(min_velocity, max_velocity, 10):
                score = 0
                valid_sequence = True
                for k in range(seq_len):
                    q_idx = row_idx - seq_len + 1 + k
                    r_idx = int(j - (seq_len - 1 - k) * v)
                    if r_idx < 0 or r_idx >= R:
                        valid_sequence = False
                        break
                    score += enhanced_sim[q_idx, r_idx]
                if valid_sequence and score > best_score:
                    best_score = score
            seq_scores[j] = best_score
        return seq_scores
    
    elif seq_match_type == 'modified':
        # 1) select the window of rows:
        start_row = max(0, row_idx - seq_len + 1)
        seq = simMat[start_row : row_idx+1 , :]       # shape = (R, num_refs), R ≤ seq_len
        seq = seq.astype(np.float64)
        
        # 2) normalize each column and row of this R×N block
        mean = np.mean(seq, axis=0, keepdims=True)
        std = np.std(seq, axis=0, keepdims=True) + 1e-8
        std[std < 1e-4] = 1.0  # Prevents blowing up near-zero std columns
        seq_colnorm = (seq - mean) / std
        seq_norm = (seq_colnorm - np.mean(seq_colnorm, axis=1, keepdims=True)) / (np.std(seq_colnorm, axis=1, keepdims=True) + 1e-8)
        
        R, N = seq_norm.shape
        seq_match_row = np.zeros(N, dtype=float)
        # 2) for each column j compute trace of the k×k block ending at j
        for j in range(N):
            k = min(R, j+1)           # grow kernel up to R
            # start column index of block:
            c0 = j - k + 1
            block = seq_norm[-k:, c0:j+1]  # now guaranteed shape (k, k)
            seq_match_row[j] = np.trace(block)

        return seq_match_row
    
    else:
        raise ValueError(f"Unknown sequence matching type: {seq_match_type}")



def compute_recall_and_matrix(
    sim_matrix,
    positives_per_query,
    seq_len=1,
    mode='single_simMat',
    seq_match_type='modified',
    apply_sequence_matching_fn=None):
    """
    Parameters:
        sim_matrix: np.ndarray or List[np.ndarray]
            - If mode == 'frame', shape is (num_queries, num_refs)
            - If mode == 'patch', list of 6 matrices (one per patch), each (num_queries, num_refs)
        positives_per_query: List[List[int]]
            - Ground truth positives per query index
        seq_len: int
            - Sequence length for sequence matching
        mode: str
            - 'frame' or 'patch'
        apply_sequence_matching_fn: Callable
            - Function that applies sequence matching

    Returns:
        recall_at_1: float
        final_sim_matrix: np.ndarray of shape (num_queries, num_refs)
    """
    assert mode in ['single_simMat', 'list_simMat'], "mode must be 'frame' or 'patch'"
    assert callable(apply_sequence_matching_fn), "You must provide a valid sequence matching function"

    if mode == 'list_simMat':
        num_queries, num_refs = sim_matrix[0].shape
        combined_matrix = np.zeros((num_queries, num_refs), dtype=np.float32)
        list_simMat=[]
        for j in range(len(sim_matrix)):
            patch_sim = sim_matrix[j]
            patch_seqmatch = apply_sequence_matching_fn(patch_sim, seq_len, seq_match_type)
            list_simMat.append(patch_seqmatch)
            combined_matrix += patch_seqmatch

        final_sim_matrix = combined_matrix  # already summed over patches

    else:  # mode == 'frame'
        final_sim_matrix = apply_sequence_matching_fn(sim_matrix, seq_len, seq_match_type)

    # Compute recall@1
    correct, total = 0, 0
    for i, current_sim in enumerate(final_sim_matrix):
        if len(positives_per_query[i]) == 0:
            continue
        best_match = np.argmax(current_sim)
        correct += int(best_match in positives_per_query[i])
        total += 1

    recall_at_1 = correct / total if total > 0 else 0.0
    return list_simMat, recall_at_1, final_sim_matrix



# def general_ensemble(
#     dataset_name: str,
#     ref_name: str,
#     qry_name: str,
#     vpr_methods: list,
#     recon_methods: list,
#     time_strs: list,
#     ensemble_over: str = 'vpr',  # Options: 'vpr', 'recon', 'time', 'patch'
#     seqLen: int = 1):
#     """
#     General ensemble function. Sums similarity matrices across specified dimensions.
#     """
#     print(f"Ensembling for {dataset_name} ({ref_name} vs {qry_name}), ensemble_over={ensemble_over}, seqLen={seqLen}")
#     suffix = '2_2' if ensemble_over == 'patch' else '1_1'
#     mode = 'patch' if ensemble_over == 'patch' else 'frame'
#     smat_list = []
#     min_rows, min_cols = None, None
#     positives_per_query = None
#     count = 0  # number of matrices summed

#     def subsample_smat(smat, step):
#         nonlocal min_rows, min_cols
#         rows = np.arange(0, smat.shape[0], step)
#         cols = np.arange(0, smat.shape[1], step)
#         smat = smat[np.ix_(rows, cols)]
#         min_rows = smat.shape[0] if min_rows is None else min(min_rows, smat.shape[0])
#         min_cols = smat.shape[1] if min_cols is None else min(min_cols, smat.shape[1])
#         return smat

#     if ensemble_over == 'patch':
#         time_res = float(time_strs[0])
#         step = int(1 / time_res)
#         base_dir = Path(f"logs/{dataset_name}/fixed_timebins_{time_strs[0]}/")
#         dataset_folder = 'BrisbaneEvent' if dataset_name == 'Brisbane' else 'NSAVP'
#         work_dir = f'../data/{dataset_folder}/image_reconstructions/fixed_timebins_{time_strs[0]}/{recon_methods[0]}'
#         queries_folder = f"{work_dir}/{qry_name}"
#         database_folder = f"{work_dir}/{ref_name}"
#         test_ds = TestDataset(database_folder, queries_folder,positive_dist_threshold=25, image_size=None, use_labels=True)
#         positives_per_query = test_ds.get_positives()

#         filename = f"{ref_name}_vs_{qry_name}_{vpr_methods[0]}_l2_reconstruct_{recon_methods[0]}_{time_strs[0]}_{mode}_{suffix}.npy"
#         file_path = base_dir / filename
#         if not file_path.exists():
#             print(f"[!] Missing: {file_path}")
#         smat_list = np.load(file_path)
#     else:
#         for vpr_method in vpr_methods if 'vpr' in ensemble_over else [vpr_methods[0]]:
#             for recon_method in recon_methods if 'recon' in ensemble_over else [recon_methods[0]]:
#                 for time_str in time_strs if 'time' in ensemble_over else [time_strs[0]]:
#                     time_res = float(time_str)
#                     step = int(1 / time_res)
#                     base_dir = Path(f"logs/{dataset_name}/fixed_timebins_{time_str}/")
#                     dataset_folder = 'BrisbaneEvent' if dataset_name == 'Brisbane' else 'NSAVP'
#                     work_dir = f'../data/{dataset_folder}/image_reconstructions/fixed_timebins_{time_str}/{recon_method}'
#                     queries_folder = f"{work_dir}/{qry_name}"
#                     database_folder = f"{work_dir}/{ref_name}"

#                     test_ds = TestDataset(
#                         database_folder, queries_folder,
#                         positive_dist_threshold=25, image_size=None, use_labels=True)
#                     positives_per_query = test_ds.get_positives()

#                     filename = f"{ref_name}_vs_{qry_name}_{vpr_method}_l2_reconstruct_{recon_method}_{time_str}_{mode}_{suffix}.npy"
#                     file_path = base_dir / filename
#                     if not file_path.exists():
#                         print(f"[!] Missing: {file_path}")
#                         continue

#                     smat = np.load(file_path)

#                     smat = subsample_smat(smat, step)
#                     smat_list.append(smat)
#                     count += 1

#     # Check if matrices have different shapes and crop if needed
#     if len(smat_list) > 1:
#         shapes = [s.shape for s in smat_list]
#         if not all(shape == shapes[0] for shape in shapes):
#             smat_list = [s[:min_rows, :min_cols] for s in smat_list] # Crop to smallest shape

#     # Final recall
#     recall, combinedSimMat = compute_recall_and_matrix(
#         smat_list, positives_per_query,
#         seq_len=seqLen, mode='list_simMat',
#         seq_match_type='modified',
#         apply_sequence_matching_fn=apply_sequence_matching
#     )
#     print(f"\n\n\n{count} ensembles with recall@1: {recall:.4f} for {dataset_name} ({ref_name} vs {qry_name}), ensemble_over={ensemble_over}, seqLen={seqLen}")

#     return combinedSimMat, recall


def general_ensemble(
    ref_name: str,
    qry_name: str,
    vpr_methods: list,
    recon_methods: list,
    time_strs: list,
    ensemble_over: str = 'vpr',  # Options: 'vpr', 'recon', 'time', 'patch'
    seqLen: int = 1):
    """
    General ensemble function. Sums similarity matrices across specified dimensions.
    """
    if ref_name.startswith('R0_') or qry_name.startswith('R0_'):
        dataset_name = 'NSAVP'
    else:
        dataset_name = 'Brisbane'

    print(f"Ensembling for {dataset_name} ({ref_name} vs {qry_name}), ensemble_over={ensemble_over}, seqLen={seqLen}")
    suffix = '2_2' if ensemble_over == 'patch' else '1_1'
    mode = 'patch' if ensemble_over == 'patch' else 'frame'
    smat_list = []
    min_rows, min_cols = None, None
    count = 0  # number of matrices summed

    def subsample_smat(smat, step):
        # nonlocal min_rows, min_cols
        # rows = np.arange(0, smat.shape[0], step)
        # cols = np.arange(0, smat.shape[1], step)
        # smat = smat[np.ix_(rows, cols)]
        # min_rows = smat.shape[0] if min_rows is None else min(min_rows, smat.shape[0])
        # min_cols = smat.shape[1] if min_cols is None else min(min_cols, smat.shape[1])
        return smat

    
    
    
    if ensemble_over == 'patch':
        time_res = float(time_strs[0])
        step = int(1 / time_res)
        dataset_folder = 'BrisbaneEvent' if dataset_name == 'Brisbane' else 'NSAVP'
        work_dir = f'../data/{dataset_folder}/image_reconstructions/fixed_timebins_1.0/{recon_methods[0]}'
        queries_folder = f"{work_dir}/{qry_name}"
        database_folder = f"{work_dir}/{ref_name}"
        test_ds = TestDataset(database_folder, queries_folder, positive_dist_threshold=25, image_size=None, use_labels=True)
        positives_per_query = test_ds.get_positives()

        base_dir = Path(f"logs/{dataset_name}/fixed_timebins_{time_strs[0]}/")
        filename = f"{ref_name}_vs_{qry_name}_{vpr_methods[0]}_l2_reconstruct_{recon_methods[0]}_{time_strs[0]}_{mode}_{suffix}.npy"
        file_path = base_dir / filename
        if not file_path.exists():
            print(f"[!] Missing: {file_path}")
        loaded_sims = np.load(file_path)
        smat_list = [subsample_smat(load_sim, step) for load_sim in loaded_sims]   # Subsample the similarity matrix
    else:
        for vpr_method in vpr_methods if 'vpr' in ensemble_over else [vpr_methods[0]]:
            for recon_method in recon_methods if 'recon' in ensemble_over else [recon_methods[0]]:
                for time_str in time_strs if 'time' in ensemble_over else [time_strs[0]]:
                    time_res = float(time_str)
                    step = int(1 / time_res)
                    
                    
                    base_dir = Path(f"logs/{dataset_name}/fixed_timebins_{time_str}/")
                    filename = f"{ref_name}_vs_{qry_name}_{vpr_method}_l2_reconstruct_{recon_method}_{time_str}_{mode}_{suffix}.npy"
                    file_path = base_dir / filename
                    if not file_path.exists():
                        print(f"[!] Missing: {file_path}")
                        continue
                    smat = np.load(file_path)
                    smat = subsample_smat(smat, step)  
                    smat_list.append(smat)
                    count += 1

                    dataset_folder = 'BrisbaneEvent' if dataset_name == 'Brisbane' else 'NSAVP'
                    work_dir = f'../data/{dataset_folder}/image_reconstructions/fixed_timebins_{time_res}/{recon_method}'
                    queries_folder = f"{work_dir}/{qry_name}"
                    database_folder = f"{work_dir}/{ref_name}"
                    test_ds = TestDataset(database_folder, queries_folder, positive_dist_threshold=25, image_size=None, use_labels=True)
                    positives_per_query = test_ds.get_positives()
    # Check if matrices have different shapes and crop if needed
    # if min_rows is not None:
    #     min_rows = len(positives_per_query) if len(positives_per_query) < min_rows else min_rows
    # print(f"Minimum rows: {min_rows}, Minimum cols: {min_cols}")
    # if len(smat_list) > 1 and (min_rows is not None and min_cols is not None):
    #     shapes = [s.shape for s in smat_list]
    #     smat_list = [s[:min_rows, :min_cols] for s in smat_list] # Crop to smallest shape
    # positives_per_query = positives_per_query[:min_rows] 
    print(f"Shape of combined similarity matrix: {smat_list[0].shape}")
    print(f"Number of queries: {len(positives_per_query)}")
    list_simMat, recall, combinedSimMat = compute_recall_and_matrix(
        smat_list, positives_per_query,
        seq_len=seqLen, mode='list_simMat',
        seq_match_type='modified',
        apply_sequence_matching_fn=apply_sequence_matching)
    
    print(f"\n\n\n{count} ensembles with recall@1: {recall:.4f} for {dataset_name} ({ref_name} vs {qry_name}), ensemble_over={ensemble_over}, seqLen={seqLen}")
    print(f"Shape of combined similarity matrix: {combinedSimMat.shape}")
    print(f"Number of queries: {len(positives_per_query)}")

    return list_simMat, positives_per_query, combinedSimMat, recall


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    """Load existing results if file exists, otherwise return empty DataFrame."""
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: Could not load existing CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def normalize_field(val):
    """Standardize a CSV field: strip, sort comma-lists, lowercase"""
    if pd.isna(val) or val == '':
        return ''
    val = str(val).strip().lower()
    if ',' in val:
        return ','.join(sorted(part.strip() for part in val.split(',')))
    return val


def is_duplicate(row: dict, existing_df: pd.DataFrame) -> bool:
    if existing_df.empty:
        return False

    for _, existing_row in existing_df.iterrows():
        match = True
        for key in ['ref_name', 'qry_name', 'ensemble_over', 'vpr_methods', 'recon_methods', 'time_strs']:
            if key not in existing_row or normalize_field(existing_row[key]) != normalize_field(row[key]):
                match = False
                break
        if match:
            return True
    return False


def save_results(results: List[Dict], csv_path: Path, append: bool = True):
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    df = pd.DataFrame(results)
    
    if append and csv_path.exists():
        # Append to existing file
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # Create new file or overwrite
        df.to_csv(csv_path, index=False)
    
    print(f"✅ {'Appended' if append and csv_path.exists() else 'Saved'} {len(results)} results to {csv_path}")


def ensemble_grid_search():
    """
    Cleaner grid search for ensemble combinations with proper error handling
    and efficient duplicate checking.
    """
    csv_path = Path('./hpc/ablate_ensemble_combinations.csv')
    results = []
    seqLen=1
    # Configuration
    seq_id_to_name = {0:'night', 1:'morning', 2: 'sunrise', 3:'sunset1', 4:'sunset2', 5:'daytime', 
                  6: 'R0_FA0', 7: 'R0_FS0', 8: 'R0_FN0', 9: 'R0_RA0', 10: 'R0_RS0', 11: 'R0_RN0'}

    # experiment_pairs = [ (3,0),(3, 1), (3, 2),(3, 4), (3, 5)] if args_cli.dataset_type == 'Brisbane' else [(6, 7),(6, 8), (8, 7), (9,11), (10, 11)]
    
    # dataset_name = "Brisbane"  # or "NSAVP"
    # ref_qry_pairs = [
    #     ('sunset1', 'night'),
    #     ('sunset1', 'morning'),
    #     ('sunset1', 'sunrise'),
    #     ('sunset1', 'sunset2'),
    #     ('sunset1', 'daytime'),
    # ]
    dataset_name = "NSAVP"
    ref_qry_pairs = [
        ('R0_FA0', 'R0_FS0'),
        ('R0_FA0', 'R0_FN0'),
        ('R0_FN0', 'R0_FS0'),
        ('R0_RA0', 'R0_RN0'),
        ('R0_RS0', 'R0_RN0'),
    ]

    vpr_all = ['mixvpr', 'megaloc', 'cosplace', 'netvlad']
    recon_all = ['RGB_camera','e2vid', 'eventCount', 'eventCount_noPolarity', 'timeSurface']
    time_all = [1.0]
    ensemble_groups = ['vpr', 'recon', 'time', 'patch']


    # Load existing results for duplicate checking
    existing_df = load_existing_results(csv_path)
    print(f"Loaded {len(existing_df)} existing results")
    
    # Create CSV with headers if it doesn't exist
    if not csv_path.exists():
        # Create empty CSV with headers
        sample_row = {
            'ref_name': '', 'qry_name': '', 'ensemble_over': '', 
            'vpr_methods': '', 'recon_methods': '', 'time_strs': '', 'recall@1': 0.0
        }
        pd.DataFrame([sample_row]).iloc[0:0].to_csv(csv_path, index=False)  # Empty df with headers

    # Run all combinations
    total_combinations = 0
    processed_combinations = 0
    
    for ref_name, qry_name in ref_qry_pairs:
        print(f"\n--- Processing {ref_name} vs {qry_name} ---")
        
        for ensemble_over in ensemble_groups:
            # Determine which parameters to vary based on ensemble_over
            if ensemble_over == 'vpr':
                # Ensemble over VPR methods: pass ALL VPR methods, single recon/time
                vpr_subsets = [vpr_all] # Full ensemble + all subsets
                recon_subsets = [[r] for r in recon_all]  # Each recon method individually
                time_subsets = [[t] for t in time_all]    # Each time value individually
            elif ensemble_over == 'recon':
                # Ensemble over reconstruction methods: pass ALL recon methods, single vpr/time
                vpr_subsets = [[v] for v in vpr_all]      # Each VPR method individually
                recon_subsets = [recon_all]  # Full ensemble + all subsets
                time_subsets = [[t] for t in time_all]    # Each time value individually
            elif ensemble_over == 'time':
                # Ensemble over time values: pass ALL time values, single vpr/recon
                vpr_subsets = [[v] for v in vpr_all]      # Each VPR method individually
                recon_subsets = [[r] for r in recon_all]  # Each recon method individually
                time_subsets = [time_all]  # Full ensemble + all subsets
            elif ensemble_over == 'patch':
                # For patch ensemble, only test individual combinations (no ensembling)
                vpr_subsets = [[v] for v in vpr_all]
                recon_subsets = [[r] for r in recon_all]
                time_subsets = [[t] for t in time_all]
            else:
                print(f"Unknown ensemble_over: {ensemble_over}")
                continue

            combos = list(itertools.product(vpr_subsets, recon_subsets, time_subsets))
            total_combinations += len(combos)
            
            for vprs, recons, times in tqdm(combos, 
                                          desc=f"{ref_name} vs {qry_name} over {ensemble_over}"):
                
                # Create result row for duplicate checking
                result_row = {
                    'ref_name': ref_name,
                    'qry_name': qry_name,
                    'ensemble_over': ensemble_over,
                    'vpr_methods': ','.join(vprs),
                    'recon_methods': ','.join(recons),
                    'time_strs': ','.join(map(str, times)),
                    'seqLen': seqLen,  # Default seqLen, can be adjusted later
                }
                
                # Check for duplicates
                if is_duplicate(result_row, existing_df):
                    print(f"Skipping duplicate: {result_row}")
                    continue
                
                try:
                    # Convert time values to strings for the function call
                    time_strs = [str(t) for t in times]
                    
                    smat, recall = general_ensemble(
                        ref_name=ref_name,
                        qry_name=qry_name,
                        vpr_methods=vprs,
                        recon_methods=recons,
                        time_strs=time_strs,
                        ensemble_over=ensemble_over,
                        seqLen=seqLen
                    )

                    # Add recall to result row
                    result_row['recall@1'] = recall
                    results.append(result_row)
                    processed_combinations += 1
                    
                    # Save periodically to avoid data loss
                    if len(results) % 10 == 0:
                        save_results(results, csv_path, append=True)
                        results = []  # Clear results list
                        existing_df = load_existing_results(csv_path)  # Reload for duplicate checking

                except Exception as e:
                    print(f"[!] Error for {ref_name} vs {qry_name}, "
                          f"ensemble={ensemble_over}, vpr={vprs}, "
                          f"recon={recons}, time={times}")
                    print(f"    Error: {e}")
                    continue

    # Save any remaining results
    if results:
        save_results(results, csv_path, append=True)

    print(f"\n🎉 Grid search completed!")
    print(f"Total combinations: {total_combinations}")
    print(f"Processed combinations: {processed_combinations}")
    print(f"Skipped combinations: {total_combinations - processed_combinations}")


if __name__ == "__main__":
    args = parse_args()

    # Set up result row
    result_row = {
        'ref_name': args.ref_seq,
        'qry_name': args.qry_seq,
        'ensemble_over': args.ensemble_over,
        'vpr_methods': args.vpr_methods,
        'recon_methods': args.recon_methods,
        'time_strs': args.time_strs,
        'seqLen': args.seq_len,}

    csv_path = Path(f'./hpc/ablate_ensemble_combination_sLen_inclRGB.csv')
    existing_df = load_existing_results(csv_path)

    if is_duplicate(result_row, existing_df):
        print("✅ Skipping — already exists in CSV.")
    else:
        list_simMat, positives_per_query, combinedSimMat, recall = general_ensemble(
            ref_name=args.ref_seq,
            qry_name=args.qry_seq,
            vpr_methods=args.vpr_methods.split(','),
            recon_methods=args.recon_methods.split(','),
            time_strs=args.time_strs.split(','),
            ensemble_over=args.ensemble_over,
            seqLen=args.seq_len
        )

        result_row['recall@1'] = recall
        save_results([result_row], csv_path, append=True)
