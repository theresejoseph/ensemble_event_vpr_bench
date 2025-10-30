
import numpy as np
from pathlib import Path
from vpr_methods_evaluation.test_dataset import TestDataset
from scipy.ndimage import uniform_filter1d
import itertools
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import argparse
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import os

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
    return recall_at_1, final_sim_matrix



def compute_recall_at_1(similarity_matrix, positives_per_query):
        correct = 0
        total = len(positives_per_query)
        for q_idx, positives in enumerate(positives_per_query):
            if len(positives) == 0:
                continue
            predicted = np.argmax(similarity_matrix[q_idx])
            if predicted in positives:
                correct += 1
        return correct / total if total > 0 else 0.0



def plot_ensemble_similarity_matrices(
    smat_list,
    fused_smat,
    positives_per_query,
    plotname):
    """
    Plot 4 ensemble member similarity matrices and the fused matrix.
    Automatically names and saves the plot based on ensemble settings.
    """
    if len(smat_list) != 4:
        raise ValueError("This plotting layout assumes exactly 4 ensemble members.")

    # Compute individual recalls
    member_recalls = [compute_recall_at_1(smat, positives_per_query) for smat in smat_list]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, (row, col) in enumerate(positions):
        ax = fig.add_subplot(gs[row, col])
        mat = smat_list[i]
        ax.imshow(mat, aspect='auto', cmap='Blues')
        ax.set_title(f"Member {i+1}\n(recall@1 = {member_recalls[i]:.4f})")
        ax.set_xlabel("Database")
        ax.set_ylabel("Query")
        ax.grid(False)

        if positives_per_query is not None:
            for q_idx, positives in enumerate(positives_per_query):
                if len(positives) == 0:
                    continue
                ax.plot(positives, [q_idx] * len(positives), 'yo', markersize=1, alpha=0.1)
                member_pred = np.argmax(mat[q_idx])
                ax.plot(member_pred, q_idx, 'x', color='#006400' if member_pred in positives else "#710E01", markersize=2, alpha=1)

    # Fused plot
    ax_fused = fig.add_subplot(gs[:, 2])
    ax_fused.imshow(fused_smat, aspect='auto', cmap='Blues')
    ax_fused.set_title(f"Fused (recall@1 = {compute_recall_at_1(fused_smat, positives_per_query):.4f})")
    ax_fused.set_xlabel("Database")
    ax_fused.set_ylabel("Query")
    ax_fused.grid(False)

    if positives_per_query is not None:
        for q_idx, positives in enumerate(positives_per_query):
            if len(positives) == 0:
                continue
            ax_fused.plot(positives, [q_idx] * len(positives), 'yo', markersize=1, alpha=0.3)
            fused_pred = np.argmax(fused_smat[q_idx])
            ax_fused.plot(fused_pred, q_idx, 'x', color='#006400' if fused_pred in positives else "#710E01", markersize=2, alpha=1)

    plt.tight_layout()
    os.makedirs('./plots/simMats', exist_ok=True)
    save_path = f'./plots/simMats/{plotname}.png'
    plt.savefig(save_path, bbox_inches='tight', format='png')
    print(f"✅ Saved similarity matrix plot to {save_path}")



def find_crop_start_index(utms, min_dist_m=2.0):
    utms = np.array(utms)
    for i in range(len(utms) - 1):
        dist = np.linalg.norm(utms[i+1] - utms[i])
        if dist >= min_dist_m:
            return i
    return 0  # fallback: no such pair found



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
        for key in ['ref_name', 'qry_name', 'ensemble_over', 'vpr_methods', 'recon_methods', 'time_strs', 'seqLen']:
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



def general_ensemble(
    ref_name: str,
    qry_name: str,
    vpr_methods: list,
    recon_methods: list,
    time_strs: list,
    ensemble_over: str = 'vpr',  # Options: 'vpr', 'recon', 'time', 'patch'
    seqLen: int = 1,
    result_row: dict = None,
    csv_path: Path = None):
    import itertools

    if ref_name.startswith('R0_') or qry_name.startswith('R0_'):
        dataset_name = 'NSAVP'
    else:
        dataset_name = 'Brisbane'

    print(f"Ensembling for {dataset_name} ({ref_name} vs {qry_name}), ensemble_over={ensemble_over}, seqLen={seqLen}")
    suffix = '2_2' if ensemble_over == 'patch' else '1_1'
    mode = 'patch' if ensemble_over == 'patch' else 'frame'
    smat_list = []
    min_rows, min_cols = None, None
    count = 0

    def subsample_smat(smat, step):
        nonlocal min_rows, min_cols
        rows = np.arange(0, smat.shape[0], step)
        cols = np.arange(0, smat.shape[1], step)
        smat = smat[np.ix_(rows, cols)]
        min_rows = smat.shape[0] if min_rows is None else min(min_rows, smat.shape[0])
        min_cols = smat.shape[1] if min_cols is None else min(min_cols, smat.shape[1])
        return smat

    def loadAndAppendSimMats(smat_list, time_str, ref_name, qry_name, vpr_method, recon_method, count):
        

        return None

    
    if 'all' in ensemble_over:
        for vpr_method in vpr_methods:
            for recon_method in recon_methods:
                for time_str in time_strs:
                    time_res = float(time_str)
                    step = int(1 / time_res)
                    base_dir = Path(f"logs/{dataset_name}/fixed_timebins_{time_str}/")
                    filename = f"{ref_name}_vs_{qry_name}_{vpr_method}_l2_reconstruct_{recon_method}_{time_str}_{mode}_{suffix}.npy"
                    file_path = base_dir / filename
                    if not file_path.exists():
                        print(f"[!] Missing: {file_path}")
                        return

                    dataset_folder = 'BrisbaneEvent' if dataset_name == 'Brisbane' else 'NSAVP'
                    work_dir = f'../data/{dataset_folder}/image_reconstructions/fixed_timebins_{time_res}/{recon_method}'
                    queries_folder = f"{work_dir}/{qry_name}"
                    database_folder = f"{work_dir}/{ref_name}"
                    test_ds = TestDataset(
                        database_folder, queries_folder,
                        positive_dist_threshold=25, image_size=None, use_labels=True)
                    r_move=find_crop_start_index(test_ds.database_utms)
                    q_move=find_crop_start_index(test_ds.queries_utms)
                    test_ds.queries_utms = test_ds.queries_utms[::step]  # Subsample queries
                    test_ds.database_utms = test_ds.database_utms[::step]
                    print(f"len(test_ds.queries_utms)={len(test_ds.queries_utms)}, len(test_ds.database_utms)={len(test_ds.database_utms)}")
                    positives_per_query = test_ds.get_positives()

                    smat = np.load(file_path) # Load and crop similarity matrix
                    smat = subsample_smat(smat, step)
                    smat_list.append(smat)
                    count += 1
                    
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

                    dataset_folder = 'BrisbaneEvent' if dataset_name == 'Brisbane' else 'NSAVP'
                    work_dir = f'../data/{dataset_folder}/image_reconstructions/fixed_timebins_{time_res}/{recon_method}'
                    queries_folder = f"{work_dir}/{qry_name}"
                    database_folder = f"{work_dir}/{ref_name}"
                    test_ds = TestDataset(
                        database_folder, queries_folder,
                        positive_dist_threshold=25, image_size=None, use_labels=True)
                    r_move=find_crop_start_index(test_ds.database_utms)
                    q_move=find_crop_start_index(test_ds.queries_utms)
                    test_ds.queries_utms = test_ds.queries_utms[::step]  # Subsample queries
                    test_ds.database_utms = test_ds.database_utms[::step]
                    print(f"len(test_ds.queries_utms)={len(test_ds.queries_utms)}, len(test_ds.database_utms)={len(test_ds.database_utms)}")
                    positives_per_query = test_ds.get_positives()

                    smat = np.load(file_path) # Load and crop similarity matrix
                    smat = subsample_smat(smat, step)
                    smat_list.append(smat)
                    count += 1

    if min_rows is not None:
        min_rows = len(positives_per_query) if len(positives_per_query) < min_rows else min_rows
    if len(smat_list) > 1 and (min_rows is not None and min_cols is not None):
        smat_list = [s[:min_rows, :min_cols] for s in smat_list]
    positives_per_query = positives_per_query[:min_rows]
    print(f"Shape of combined similarity matrix: {smat_list[0].shape}")
    print(f"Number of queries: {len(positives_per_query)}")
    

    # Evaluate individual recalls
    individual_recalls = []
    individual_smat_list = []
    for idx, simMat in enumerate(smat_list):
        recall_single, simMat_withseq = compute_recall_and_matrix(
            [simMat], positives_per_query, seq_len=seqLen,
            mode='list_simMat', seq_match_type='modified',
            apply_sequence_matching_fn=apply_sequence_matching)
        individual_recalls.append(recall_single)
        individual_smat_list.append(simMat_withseq)

    # Compute disagreement between matrices
    disagreements = []
    for A, B in itertools.combinations(smat_list, 2):
        diff_norm = np.linalg.norm(A - B)
        avg_norm = np.linalg.norm(A + B)
        disagreements.append(diff_norm / avg_norm if avg_norm > 0 else 0)
    avg_disagreement = float(np.mean(disagreements)) if disagreements else 0.0

    # Ensemble matrices
    mean_smat = np.mean(individual_smat_list, axis=0)
    max_smat = np.max(individual_smat_list, axis=0)
    med_smat = np.median(np.array(individual_smat_list), axis=0)
    if len(individual_smat_list) == 4:
        fused_smat = mean_smat.copy()
        vpr_str = '-'.join(vpr_methods)
        time_str = '-'.join(time_strs)
        recon_str = '-'.join(recon_methods)
        plotname= f"{ref_name}_vs_{qry_name}_L{seqLen}_VPR-{vpr_str}_T-{time_str}_R-{recon_str}".replace('/', '-')
        plot_ensemble_similarity_matrices(individual_smat_list, fused_smat, positives_per_query, plotname )

    # Evaluate ensemble fusions
    recall_mean = compute_recall_at_1(mean_smat, positives_per_query)
    recall_max = compute_recall_at_1(max_smat, positives_per_query)
    recall_median = compute_recall_at_1(med_smat, positives_per_query)

    print(f"\n{count} members — Recall@1 (mean): {recall_mean:.4f}, max: {recall_max:.4f}, median: {recall_median:.4f}, disagreement: {avg_disagreement:.4f}")
    
    if result_row is not None:
        result_row.update({
            'recall@1_mean': recall_mean,
            'recall@1_max': recall_max,
            'recall@1_median': recall_median,
            'recall@1_individual': str(individual_recalls),
            'avg_disagreement': avg_disagreement,
        })
        if csv_path:
            save_results([result_row], csv_path, append=True)

    

if __name__ == "__main__":
    args = parse_args()
    result_row = {
        'ref_name': args.ref_seq,
        'qry_name': args.qry_seq,
        'ensemble_over': args.ensemble_over,
        'vpr_methods': args.vpr_methods,
        'recon_methods': args.recon_methods,
        'time_strs': args.time_strs,
        'seqLen': args.seq_len,}

    csv_path = Path(f'./hpc/ablate_ensemble_combination_additionalResults.csv')
    existing_df = load_existing_results(csv_path)

    if is_duplicate(result_row, existing_df):
        print("✅ Skipping — already exists in CSV.")
    else:
        general_ensemble(
        ref_name=args.ref_seq,
        qry_name=args.qry_seq,
        vpr_methods=args.vpr_methods.split(','),
        recon_methods=args.recon_methods.split(','),
        time_strs=args.time_strs.split(','),
        ensemble_over=args.ensemble_over,
        seqLen=args.seq_len,
        result_row=result_row,
        csv_path=csv_path)

# python ablate_ensembles.py --dataset_name NSAVP --ref_seq R0_FN0 --qry_seq R0_FS0 --vpr_methods "mixvpr,megaloc,cosplace,netvlad" --recon_methods "e2vid,eventCount,eventCount_noPolarity,timeSurface" --time_strs "0.1,0.25,0.5,1.0" --ensemble_over all --seq_len 10
