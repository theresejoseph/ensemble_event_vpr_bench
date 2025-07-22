import sys
from pathlib import Path
from glob import glob
import time  # Make sure this is at the top of your file
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
import parse as parse
import sys

# import faiss
import numpy as np
import torch                      
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

# import visualizations
from test_dataset import TestDataset

import os
import csv
import pandas as pd

''' ------------------------------------------------   ALL VARIABLES ------------------------------------------------

sequences_BrisbaneEvent=["night","morning", "sunrise", "sunset1", "sunset2", "daytime"] #Brisbane event dataset 
sequences_MichiganNSAVP= ["R0_FA0", "R0_FS0", "R0_FN0", "R0_RA0", "R0_RS0", "R0_RN0"] # NSAVP dataset (R0=route0, F=forward, R=reverse, A=Afternoon, S=Sunset, N=Night)
representations=['EventSignedCount_', 'EventUnsignedCount_', 'EventFrequency_', 'TimeSurface_']
methods=['mixvpr','convap', 'cosplace', 'sfrs', 'eigenplaces', 'dinomix' 'netvlad', 'megaloc'] #'apgem', 'salad', 'cricavpr']
sequence_matching_lengths=[1,5,10,15,25,50,75]
time_res = [0.1, 0.2, 0.5, 1.0]

------------------------------------------------------------------------------------------------------------------ '''


# def split_into_patches(imgs, args):
#     """
#     Split a batch of images into patches.

#     Returns:
#         patches (List[torch.Tensor]) or None if patches are too small
#     """
#     B, C, H, W = imgs.shape
#     patches = []

#     min_patch_size = 14 # megaloc requires larger patches

#     if args.grid_or_nest == 'grid':
#         num_rows = args.patch_num_rows
#         num_cols = args.patch_num_cols
#         patch_H = H // num_rows
#         patch_W = W // num_cols

#         if patch_H < min_patch_size or patch_W < min_patch_size:
#             print(f"❌ Skipping config: patch size too small ({patch_H}x{patch_W})")
#             return None

#         for r in range(num_rows):
#             for c in range(num_cols):
#                 patch = imgs[:, :, r*patch_H:(r+1)*patch_H, c*patch_W:(c+1)*patch_W]
#                 patches.append(patch)

#     elif args.grid_or_nest == 'nest':
#         scale_factor = args.nest_scale_factor
#         for i in range(args.num_patches):
#             scale = scale_factor ** i
#             crop_H = int(H * scale)
#             crop_W = int(W * scale)

#             if crop_H < min_patch_size or crop_W < min_patch_size:
#                 print(f"❌ Skipping nested patch {i}: too small ({crop_H}x{crop_W})")
#                 return None

#             start_H = (H - crop_H) // 2
#             start_W = (W - crop_W) // 2
#             patch = imgs[:, :, start_H:start_H+crop_H, start_W:start_W+crop_W]
#             patches.append(patch)
#     else:
#         raise ValueError(f"Unknown patch mode: {args.grid_or_nest}")

#     return patches

def split_into_patches(imgs, args):
    """
    Split a batch of images into patches.
    
    Args:
        imgs (torch.Tensor): (B, C, H, W)
        num_rows (int): Number of rows to split into
        num_cols (int): Number of columns to split into
    
    Returns:
        patches (List[torch.Tensor]): List of length num_rows*num_cols,
            each of shape (B, C, patch_H, patch_W)
    """
    B, C, H, W = imgs.shape
    patch_H = H // args.patch_num_rows
    patch_W = W // args.patch_num_cols

    patches = []
    for r in range(args.patch_num_rows):
        for c in range(args.patch_num_cols):
            patch = imgs[:, :, r*patch_H:(r+1)*patch_H, c*patch_W:(c+1)*patch_W]
            patches.append(patch)
    
    return patches  

def extract_patch_descriptors(args, model, img_batch, device):
    """
    Extract descriptors for each patch of the images.

    Args:
        model: VPR model (torch.nn.Module)
        img_batch: (B, C, H, W) tensor
        device: torch.device

    Returns:
        patch_descs (List[np.ndarray]): List of length 6, each with shape (B, descriptor_dim)
    """
    model.eval()
    with torch.no_grad():
        patches = split_into_patches(img_batch, args)  
        if patches is None:
            print("⚠ Patches too small, skipping this config.")
            return None, None  # Or raise a handled exception
        patch_descs = []
        for patch in patches:
            desc = model(patch.to(device))  # (B, descriptor_dim)
            assert desc.shape[1] == args.descriptors_dimension, "Descriptor dimension mismatch!"
            patch_descs.append(desc.cpu().numpy())
    return patch_descs, patches 



def compute_similarity_matrices(simMatPath, args, metric):
    """
    Compute similarity matrices per patch for patch-based VPR.
    
    Parameters:
        qry_patch_descs (List[np.ndarray]): List of length 6. Each element is (num_queries, descriptor_dim).
        db_patch_descs (List[np.ndarray]): List of length 6. Each element is (num_refs, descriptor_dim).
        metric (str): Either 'cosine' or 'l2'.

    Returns:
        simMats (List[np.ndarray]): List of 6 similarity matrices (num_queries, num_refs).
    """
    
    start_sim = time.time()
    if not os.path.exists(simMatPath):
        if args.method == 'sad':
            queries_descriptors, database_descriptors = load_desc_sad(args)
        else:
            queries_descriptors, database_descriptors = load_desc(args)
        if queries_descriptors is None or database_descriptors is None:
            print("⚠ Failed to load descriptors, skipping similarity matrix computation.")
            return None, None  

        if args.patch_or_frame == 'patch':
            S = []
            for i in range(args.num_patches):
                qry_descs = queries_descriptors[i]  # (num_queries, descriptor_dim)
                db_descs = database_descriptors[i]    # (num_refs, descriptor_dim)
                if args.method == 'sad':
                    simMat = -np.sum(np.abs(qry_descs[:, None] - db_descs[None, :]), axis=2)
                elif metric == 'cosine':
                    Q = qry_descs / (np.linalg.norm(qry_descs, axis=1, keepdims=True) + 1e-8)
                    D = db_descs / (np.linalg.norm(db_descs, axis=1, keepdims=True) + 1e-8)
                    simMat = Q @ D.T
                elif metric == 'l2':
                    q2 = np.sum(qry_descs ** 2, axis=1, keepdims=True)
                    d2 = np.sum(db_descs ** 2, axis=1, keepdims=True).T
                    simMat = 2 * (qry_descs @ db_descs.T) - q2 - d2
                else:
                    raise ValueError("Unsupported metric: choose 'cosine' or 'l2'")
                S.append(simMat)
            if args.saveSimMat == True:
                np.save(simMatPath, S) 
            del queries_descriptors, database_descriptors

        elif args.patch_or_frame == 'frame':
            # weights=1
            D = database_descriptors
            Q = queries_descriptors

            if args.method == 'sad':
                S = -np.sum(np.abs(Q[:, None] - D[None, :]), axis=2)

            elif metric == 'cosine':
                # L2‐normalize then dot-product
                D_norm = D / (np.linalg.norm(D, axis=1, keepdims=True)+1e-8)
                Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True)+1e-8)
                S = Q_norm @ D_norm.T         # shape M×N

            elif metric == 'l2':
                q2 = np.sum(Q**2, axis=1, keepdims=True)   # M×1
                d2 = np.sum(D**2, axis=1, keepdims=True).T # 1×N
                S = 2*(Q @ D.T) - q2 - d2                  # M×N
            else:
                raise ValueError
            if args.saveSimMat == True:
                np.save(simMatPath, S) 
            del D, Q, database_descriptors, queries_descriptors
        
        
    else:
        print("\nLoading existing sim mat\n")
        S = np.load(simMatPath, allow_pickle=True)
    
    sim_time = time.time() - start_sim
    return S, sim_time



def load_desc_sad(args):
    """
    Load flattened image or patch descriptors for SAD (L1) distance.
    Applies average pooling before flattening to reduce memory usage.
    """
    import torch.nn.functional as F
    log_dir = Path("logs") / args.log_dir
    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )

    def downsample_im(tensor, kernel_size=8):
        return F.avg_pool2d(tensor, kernel_size=kernel_size)

    def flatten_im(tensor, eps=1e-8):
        descs = tensor.reshape(tensor.size(0), -1).cpu().numpy()
        # Per-descriptor normalization to unit norm
        norms = np.linalg.norm(descs, axis=1, keepdims=True)
        norms[norms < eps] = 1.0  # avoid divide by 0
        descs = descs / norms
        return descs.astype(np.float32)  # or float16 if safe

    if args.patch_or_frame == "patch":
        all_patch_descs = None
        num_total = len(test_ds)

        # --- DATABASE ---
        database_loader = DataLoader(
            Subset(test_ds, range(test_ds.num_database)),
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        for images, indices in database_loader:
            images=downsample_im(images)
            patches = split_into_patches(images, args)
            for i in range(args.num_patches):
                descs = flatten_im(patches[i])  
                # descs = descs/ (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8)
                if all_patch_descs is None:
                    desc_dim = descs.shape[1]
                    all_patch_descs = [np.empty((num_total, desc_dim), dtype='float32') for _ in range(args.num_patches)]
                all_patch_descs[i][indices.numpy(), :] = descs

        # --- QUERIES ---
        query_loader = DataLoader(
            Subset(test_ds, range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)),
            num_workers=args.num_workers,
            batch_size=1,
        )
        for images, indices in query_loader:
            images=downsample_im(images)
            patches = split_into_patches(images, args)
            for i in range(args.patch_num_rows*args.patch_num_cols):
                descs = flatten_im(patches[i]) 
                # descs = descs / (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8)
                all_patch_descs[i][indices.numpy(), :] = descs

        queries_descriptors = [p[test_ds.num_database:] for p in all_patch_descs]
        database_descriptors = [p[:test_ds.num_database] for p in all_patch_descs]

    elif args.patch_or_frame == "frame":
        sample_img = test_ds[0][0]
        desc_dim = flatten_im(downsample_im(sample_img.unsqueeze(0))).shape[1]
        all_descriptors = np.empty((len(test_ds), desc_dim), dtype='float32')

        # --- DATABASE ---
        database_loader = DataLoader(
            Subset(test_ds, range(test_ds.num_database)),
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        for images, indices in database_loader:
            descs = flatten_im(downsample_im((images)))  
            # descs = descs/ (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8)
            all_descriptors[indices.numpy(), :] = descs

        # --- QUERIES ---
        query_loader = DataLoader(
            Subset(test_ds, range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)),
            num_workers=args.num_workers,
            batch_size=1,
        )
        for images, indices in query_loader:
            descs = flatten_im(downsample_im((images))) 
            # descs = descs/ (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8)
            all_descriptors[indices.numpy(), :] = descs

        queries_descriptors = all_descriptors[test_ds.num_database:]
        database_descriptors = all_descriptors[:test_ds.num_database]

    else:
        raise ValueError("patch_or_frame must be either 'patch' or 'frame'.")

    if args.save_descriptors:
        print(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    return queries_descriptors, database_descriptors



def load_desc(args):
    from loguru import logger
    import vpr_models as vpr_models
    log_dir = Path("logs") / args.log_dir
    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,)

    

    if args.patch_or_frame == "patch":
        logger.remove()
        logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
        logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
        logger.add(log_dir / "debug.log", level="DEBUG")
        logger.info(" ".join(sys.argv))
        logger.info(f"Testing on {test_ds}")

        with torch.inference_mode():
            database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
            database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size)
            # all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
            all_patch_descs = [np.empty((len(test_ds), args.descriptors_dimension), dtype='float32') for _ in range(args.num_patches)]
            
            patch_sums = [[] for _ in range(args.num_patches)]
            patch_vars = [[] for _ in range(args.num_patches)]

            # For each batch of images
            for images, indices in database_dataloader:
                patch_descs, patches = extract_patch_descriptors(args, model, images, args.device)  # List of 6 arrays
                if patches is not None:
                    for i in range(args.num_patches):
                        all_patch_descs[i][indices.numpy(), :] = patch_descs[i]

            
            # logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
            queries_subset_ds = Subset(test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)))
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
            # For each batch of images
            for images, indices in queries_dataloader:
                patch_descs, patches = extract_patch_descriptors(args, model, images, args.device)  # List of 6 arrays
                if patches is not None:
                    for i in range(args.num_patches):
                        all_patch_descs[i][indices.numpy(), :] = patch_descs[i]
        
        if patches is None:
            logger.error("⚠ Patches too small, skipping this config.")
            return None, None
        else:
            logger.info(f"Extracted descriptors for {len(test_ds)} images with {args.num_patches} patches each.")
            logger.info(f"Descriptors shape: {[pds.shape for pds in all_patch_descs]}")
            queries_descriptors = [pds[test_ds.num_database :] for pds in all_patch_descs]
            database_descriptors = [pds[: test_ds.num_database] for pds in all_patch_descs]
        # avg_patch_activity = np.array([np.mean(p) for p in patch_sums])
        # norm_patch_activity = avg_patch_activity / np.sum(avg_patch_activity)

    elif args.patch_or_frame == "frame":
        with torch.inference_mode():
            all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")

            database_dataloader = DataLoader(
                dataset=Subset(test_ds, range(test_ds.num_database)),
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            )
            for images, indices in database_dataloader:
                descriptors = model(images.to(args.device)).cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

            queries_dataloader = DataLoader(
                dataset=Subset(test_ds, range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)),
                num_workers=args.num_workers,
                batch_size=1,
            )
            for images, indices in queries_dataloader:
                descriptors = model(images.to(args.device)).cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

        queries_descriptors = all_descriptors[test_ds.num_database:]
        database_descriptors = all_descriptors[:test_ds.num_database]
        # norm_patch_activity =1

    else:
        raise ValueError("patch_or_frame must be either 'patch' or 'frame'.")

    if args.save_descriptors:
        print(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    return queries_descriptors, database_descriptors



def seq_match_row(simMat, row_idx, seq_len, seq_match_type='modified'):
    """
    Row‑wise diagonal sequence matching using z score normalization to avoid collapse to zero.
    """
    from scipy.ndimage import uniform_filter1d
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
        seq_colnorm = (seq - np.mean(seq, axis=0, keepdims=True)) / (np.std(seq, axis=0, keepdims=True) + 1e-8)
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


def plot_simmat_with_matches(simMat, positives_per_query, save_path, patch_num='', seq_len=1, seq_match_type='modified'):
    matches = []
    seqMat=[]
    correct = 0
    total = 0
    num_queries, num_refs = simMat.shape
    seqMat = []
    matches = []
    correct = total = 0


    for i in range(num_queries):
        # compute sequence‑matched sim for row i (always done)
        if seq_len == 1:
            sim_min, sim_max = simMat[i].min(), simMat[i].max()
            current_sim = simMat[i]#(simMat[i] - sim_min) / (sim_max - sim_min + 1e-8)
        else:
            current_sim = seq_match_row(simMat, i, seq_len, seq_match_type=seq_match_type)/seq_len

        seqMat.append(current_sim)
        best = np.argmax(current_sim)
        sim_score = current_sim[best]
        is_rel = int(best in positives_per_query[i])

        matches.append((i, best, sim_score, is_rel))
        if len(positives_per_query[i])>0:
            correct += is_rel
            total += 1

    recall = correct / total if total > 0 else 0.0
    from skimage.measure import shannon_entropy
    print(f"[INFO] {patch_num} Recall: {recall:.4f} , std: {np.std(np.array(seqMat))}")


    if save_path!= None:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=np.array(seqMat), colorscale='Cividis'))

        # Plot positive markers only (no match markers)
        for q_idx, pos_refs in enumerate(positives_per_query):
            if len(pos_refs)>0:  # Only plot if positives exist
                fig.add_trace(go.Scatter(
                    x=pos_refs,
                    y=[q_idx]*len(pos_refs),
                    mode='markers',
                    marker=dict(size=5, color='violet', opacity=0.5),
                    name='Positives',
                    showlegend=False ))
        
        for display_idx, (_, best_match_id, _, is_relevant) in enumerate(matches):
            color = 'lime' if is_relevant else 'red'
            symbol = 'x' if is_relevant else 'circle'
            fig.add_trace(go.Scatter(
                x=[best_match_id],
                y=[display_idx],
                mode='markers',
                marker=dict(size=6, color=color, symbol=symbol),
                name='Matches',
                showlegend=False ))

        fig.update_layout(title=f'{patch_num} Recall: {recall:.2f}',
            xaxis_title='Reference Index',
            yaxis_title='Query Index',
            yaxis_autorange='reversed')

        # Save to interactive HTML
        save_path = Path(save_path)
        fig.write_html(str(save_path.with_name(f"{save_path.stem}{patch_num}.html")))

    return seqMat, matches



def getPRCurve_sim(similarity_matrix, positives_per_query):
    """
    Calculate PR curve given a similarity matrix instead of mInds and mSims.
    
    Args:
        similarity_matrix: (N_database, N_queries) similarity scores
        positives_per_query: list of lists, where positives_per_query[i] contains 
                           the database indices that are positive matches for query i
    
    Returns:
        prfData: numpy array of [precision, recall] pairs
        auc: area under the precision-recall curve
    """
    
    # Extract best match indices and similarities for each query
    mInds = np.argmax(similarity_matrix, axis=1)  # Best match index per query
    mSims = np.max(similarity_matrix, axis=1)     # Best match similarity per query
    
    # Ensure we have the right number of queries
    n_queries = similarity_matrix.shape[0]
    assert len(positives_per_query) == n_queries, f"positives_per_query length ({len(positives_per_query)}) != n_queries ({n_queries})"
    
    # Ensure mInds and mSims are 1D with correct length
    mInds = np.asarray(mInds).reshape(-1)
    mSims = np.asarray(mSims).reshape(-1)
    
    assert len(mInds) == n_queries, f"mInds length ({len(mInds)}) != n_queries ({n_queries})"

    # --- Recall@1 ---
    correct_at_1 = sum(
        mInds[i] in positives_per_query[i] for i in range(n_queries) if len(positives_per_query[i]) > 0
    )
    valid_queries = sum(len(p) > 0 for p in positives_per_query)
    recall_at_1 = correct_at_1 / valid_queries if valid_queries > 0 else 0.0

    
    prfData = []
    ub, lb = np.max(mSims), np.min(mSims)  # Upper and lower bounds
    
    step = (ub - lb) / 100.0
    thresholds = np.arange(lb, ub + step, step)
    
    for thresh in thresholds:
        matchFlags = mSims >= thresh  # Matches above threshold
        outVals = mInds.copy()
        outVals[~matchFlags] = -1  # Invalidate matches below threshold

        correct = 0
        total = 0
        relevant = 0

        for i in range(len(outVals)):
            if len(positives_per_query[i]) == 0:
                continue  # skip queries with no ground truth
            relevant += 1

            pred = int(outVals[i])  # Ensure it's a scalar integer
            if pred != -1:
                total += 1
                if pred in positives_per_query[i]:
                    correct += 1

        p = correct / total if total > 0 else 0.0
        r = correct / relevant if relevant > 0 else 0.0
        prfData.append([p, r])
    
    # Calculate AUC using trapezoidal rule
    if len(prfData) > 1:
        precisions = [x[0] for x in prfData]
        recalls = [x[1] for x in prfData]
        # Sort by recall for proper AUC calculation
        sorted_pairs = sorted(zip(recalls, precisions))
        sorted_recalls, sorted_precisions = zip(*sorted_pairs)
        auc = np.trapz(sorted_precisions, sorted_recalls)
    else:
        auc = 0.0
    
    return np.array(prfData), auc, recall_at_1



def store_results_to_csv(csv_path, row_dict, header):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)



def check_if_result_exists(csv_path, row_dict, key_fields):
    if not csv_path.exists():
        return False

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Match all key fields
            if all(row.get(k) == str(row_dict[k]) for k in key_fields):
                auc_val = row.get("auc", "")
                if auc_val.strip() == "" or auc_val.lower() == "nan":
                    return False  # AUC is missing → treat as incomplete
                return True  # Match found and AUC is filled
    return False



def run_vpr_save_results(args):
    """
    Run a single VPR evaluation and return Recall@1 (no CSV output, no seq_len loop).
    """
    from load_and_save import make_paths

    metric = 'l2'
    args.num_patches = args.patch_num_cols * args.patch_num_rows if args.grid_or_nest == 'grid' else args.num_patches
    seq_len = args.seq_len  # fixed
    args.positive_dist_threshold = 25
    args.rep = args.reconstruct_method_name
    method = args.method
    patch_or_frame = args.patch_or_frame
    if patch_or_frame == 'frame':
        args.patch_num_rows = 1
        args.patch_num_cols = 1
    ref_seq = args.sequences[args.idR]
    qry_seq = args.sequences[args.idQ]
    
    
    # Setup paths
    make_paths(args, ref_seq)
    args.database_folder = str(args.save_images_dir)
    num_ref_frames = len(glob(f"{args.database_folder}/**/*", recursive=True))

    make_paths(args, qry_seq)
    args.queries_folder = str(args.save_images_dir)
    num_qry_frames = len(glob(f"{args.queries_folder}/**/*", recursive=True))
     # Create log directories
    args.log_dir = f"{args.dataset_type}/{args.subfolder_dir.split('/')[-1]}"
    csv_path = Path(f"./results/vpr_results_{args.dataset_type}_{args.subfolder_dir.split('/')[-1]}.csv")
    args.time_res = None if args.count_bin == 1 else args.time_res
    simMatPath = Path("logs") / args.log_dir / f"{ref_seq}_vs_{qry_seq}_{args.method}_{metric}_reconstruct_{args.reconstruct_method_name}_{args.time_res}_{args.patch_or_frame}_{args.patch_num_rows}_{args.patch_num_cols}.npy"

    if args.idR >= 12 or args.idQ >= 12:
        csv_path = Path(f'./hpc/patch_grid_search_results1.csv')
        args.saveSimMat = False


    # check if paths are valid
    if args.save_images_dir is None:
        print(f"Failed to load data for {ref_seq} vs {qry_seq}")
        return
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return
    print(f"Got paths:")
    print(f"  Reference: {args.database_folder} ")
    print(f"  Query: {args.queries_folder} ")
    print(f"  CSV path: {csv_path}")
    print(f"  Similarity matrix path: {simMatPath}")
    
    
    csv_header = ["ref_seq", "qry_seq", "reconstruction_name", "vpr_method", 
                "seq_len", "bin_type","binning_strategy", "events_per_bin", "time_res",
                "positive_dist_thresh", "patch_or_frame", "seq_match_type", "num_qry_frames",
                "num_ref_frames","recall_at_1", "auc", "runtime", "patch_rows", "patch_cols"]
    row_dict = {
        "ref_seq": ref_seq,
        "qry_seq": qry_seq,
        "reconstruction_name": args.reconstruct_method_name,
        "vpr_method": args.method,
        "seq_len": seq_len,
        "bin_type": "countbin" if args.count_bin else "timebin",
        "binning_strategy": "adaptive" if args.adaptive_bin else "fixed",
        "events_per_bin": args.events_per_bin if args.count_bin else "",
        "time_res": args.time_res if not args.adaptive_bin and not args.count_bin else "",
        "positive_dist_thresh": args.positive_dist_threshold,
        "patch_or_frame": patch_or_frame,
        "seq_match_type": args.seq_match_type,
        "num_qry_frames": num_qry_frames,
        "num_ref_frames": num_ref_frames,
        "recall_at_1": "",  # To be filled later
        "auc": "",          # You can compute and add AUC if needed
        "runtime": "",
        "patch_rows": args.patch_num_rows,
        "patch_cols": args.patch_num_cols}       # To be filled later}

    key_fields = ["ref_seq", "qry_seq", "reconstruction_name", "vpr_method", 
                  "seq_len", "bin_type", "binning_strategy", "events_per_bin", "time_res",
                  "positive_dist_thresh", "patch_or_frame", "seq_match_type","num_qry_frames","num_ref_frames", 
                  "patch_rows", "patch_cols"]

    if check_if_result_exists(csv_path, row_dict, key_fields):
        print(f"\nSkipping experiment for {ref_seq} vs {qry_seq} with method {method} - already done.\n")
        return None

    else:
        S, sim_time = compute_similarity_matrices(simMatPath, args, metric)
        test_ds = TestDataset(args.database_folder, args.queries_folder, 
                            positive_dist_threshold=args.positive_dist_threshold, 
                            image_size=args.image_size, use_labels=args.use_labels)
        positives_per_query = test_ds.get_positives()
        print(f"Running VPR: {ref_seq} vs {qry_seq}, method={method}, rep={args.rep}, seq_len={seq_len}")
        start_match = time.time()
        
        if patch_or_frame == 'patch':
            num_queries, num_refs = S[0].shape
            combined_smats = np.zeros((num_queries, num_refs), dtype=np.float32)
            
            for j in range(args.num_patches):
                seqMat_patch, _ = plot_simmat_with_matches(S[j], positives_per_query, None, 
                                                        patch_num=f'_Patch{j}-{args.num_patches}', 
                                                        seq_len=seq_len, seq_match_type=args.seq_match_type)
                combined_smats += np.array(seqMat_patch)

            _, all_matches = plot_simmat_with_matches(combined_smats, positives_per_query, None, 
                                                    patch_num='patch_combined', seq_len=1, seq_match_type=args.seq_match_type)
            _,auc,r1 =getPRCurve_sim(np.array(combined_smats), positives_per_query)
        else:
            seqMat, all_matches = plot_simmat_with_matches(S, positives_per_query, None, 
                                                    patch_num='frame', seq_len=seq_len, seq_match_type=args.seq_match_type)
            _,auc,r1 = getPRCurve_sim(np.array(seqMat), positives_per_query)

        correct = sum(best_match_idx in positives_per_query[query_idx] 
                    for query_idx, best_match_idx, _, _ in all_matches)
        recall_at_1 = correct / len(all_matches) if len(all_matches) > 0 else 0.0
        runtime = sim_time + (time.time() - start_match)

        print(f"Recall@1 = {recall_at_1:.3f} | Runtime = {runtime:.2f} sec")
        row_dict["auc"] = f"{auc:.4f}"
        row_dict["recall_at_1"] = f"{recall_at_1:.4f}"
        row_dict["runtime"] = f"{runtime:.2f}"
        store_results_to_csv(csv_path, row_dict, csv_header)

    return recall_at_1





def run_vpr(args):
    """
    Run a single VPR evaluation and return Recall@1 (no CSV output, no seq_len loop).
    """
    from load_and_save import make_paths

    metric = 'l2'
    args.num_patches = args.patch_num_cols * args.patch_num_rows if args.grid_or_nest == 'grid' else args.num_patches
    seq_len = args.seq_len  # fixed
    args.positive_dist_threshold = 25
    args.rep = args.reconstruct_method_name
    method = args.method
    patch_or_frame = args.patch_or_frame
    ref_seq = args.sequences[args.idR]
    qry_seq = args.sequences[args.idQ]
    args.dataset_type = 'NSAVP' if 6 <= args.idR <= 11 else 'Brisbane'
    #ref
    make_paths(args, ref_seq)
    args.database_folder = str(args.save_images_dir)
    num_ref_frames = len(glob(f"{args.database_folder}/**/*", recursive=True))
    #qry
    make_paths(args, qry_seq)
    args.queries_folder = str(args.save_images_dir)
    num_qry_frames = len(glob(f"{args.queries_folder}/**/*", recursive=True))
    if args.save_images_dir is None:
        print(f"Failed to load data for {ref_seq} vs {qry_seq}")
        return None
    args.log_dir = args.subfolder_dir.split('/')[-1]

    print(f"Loaded data:")
    print(f"  Reference: {args.database_folder} ({num_ref_frames} frames)")
    print(f"  Query: {args.queries_folder} ({num_qry_frames} frames)")
    print(f"  Binning: adaptive={args.adaptive_bin}, bin_type={'count' if args.count_bin else 'time'} ")
    print(f"  Log directory: {args.log_dir}")
    
    # Create log directories
    log_dir = Path("logs") / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    simMatPath = log_dir / f"{ref_seq}_vs_{qry_seq}_{args.method}_{metric}_reconstruct_{args.reconstruct_method_name}_{args.time_res}_{args.patch_or_frame}.npy"
    print(f"Similarity matrix path: {simMatPath}")
    S, sim_time = compute_similarity_matrices(simMatPath, args, metric)
    if S is None:
        print("⚠ Skipping due to invalid similarity matrix")
        return None, None
    
    test_ds = TestDataset(args.database_folder, args.queries_folder, 
                        positive_dist_threshold=args.positive_dist_threshold, 
                        image_size=args.image_size, use_labels=args.use_labels)
    positives_per_query = test_ds.get_positives()

    if patch_or_frame == 'patch':
        num_queries, num_refs = S[0].shape
        combined_smats = np.zeros((num_queries, num_refs), dtype=np.float32)
        
        for j in range(args.num_patches):
            seqMat_patch, _ = plot_simmat_with_matches(S[j], positives_per_query, None, 
                                                    patch_num=f'_Patch{j}-{args.num_patches}', 
                                                    seq_len=seq_len, seq_match_type=args.seq_match_type)
            combined_smats += np.array(seqMat_patch)

        _, all_matches = plot_simmat_with_matches(combined_smats, positives_per_query, None, 
                                                patch_num='patch_combined', seq_len=1, seq_match_type=args.seq_match_type)
        _,auc,r1 =getPRCurve_sim(np.array(combined_smats), positives_per_query)
    else:
        seqMat, all_matches = plot_simmat_with_matches(S, positives_per_query, None, 
                                                patch_num='frame', seq_len=seq_len, seq_match_type=args.seq_match_type)
        _,auc,r1 = getPRCurve_sim(np.array(seqMat), positives_per_query)
    print(f"Running VPR: {ref_seq} vs {qry_seq}, method={method}, rep={args.rep}, seq_len={seq_len}, AUC={auc:.4f}")    
    correct = sum(best_match_idx in positives_per_query[query_idx] 
                for query_idx, best_match_idx, _, _ in all_matches)
    recall_at_1 = correct / len(all_matches) if len(all_matches) > 0 else 0.0

    return recall_at_1, auc




def run_vpr_fill_auc(args):
    from load_and_save import make_paths
    metric = 'l2'
    args.num_patches = args.patch_num_cols * args.patch_num_rows if args.grid_or_nest == 'grid' else args.num_patches
    seq_len = args.seq_len
    args.positive_dist_threshold = 25
    args.rep = args.reconstruct_method_name
    ref_seq = args.sequences[args.idR]
    qry_seq = args.sequences[args.idQ]
    patch_or_frame = args.patch_or_frame
    if patch_or_frame == 'frame':
        args.patch_num_rows = 1
        args.patch_num_cols = 1
    args.dataset_type = 'NSAVP' if args.idR >= 6 or args.idQ >= 6 else 'Brisbane'

    # Setup paths
    make_paths(args, ref_seq)
    args.database_folder = str(args.save_images_dir)
    make_paths(args, qry_seq)
    args.queries_folder = str(args.save_images_dir)
    args.log_dir = f"{args.dataset_type}/{args.subfolder_dir.split('/')[-1]}"
    csv_path = Path(f"./results/vpr_results_{args.dataset_type}_{args.subfolder_dir.split('/')[-1]}.csv")
    simMatPath = Path("logs") / args.log_dir / f"{ref_seq}_vs_{qry_seq}_{args.method}_{metric}_reconstruct_{args.reconstruct_method_name}_{args.time_res}_{args.patch_or_frame}_{args.patch_num_rows}_{args.patch_num_cols}.npy"
    # check if paths are valid
    if args.save_images_dir is None:
        print(f"Failed to load data for {ref_seq} vs {qry_seq}")
        return
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return
    print(f"Got paths:")
    print(f"  Reference: {args.database_folder} ")
    print(f"  Query: {args.queries_folder} ")
    print(f"  CSV path: {csv_path}")
    print(f"  Similarity matrix path: {simMatPath}")


    csv_df = pd.read_csv(csv_path)
    row_signature = {
        "ref_seq": ref_seq,
        "qry_seq": qry_seq,
        "reconstruction_name": args.reconstruct_method_name,
        "vpr_method": args.method,
        "seq_len": seq_len,
        "bin_type": "countbin" if args.count_bin else "timebin",
        "binning_strategy": "adaptive" if args.adaptive_bin else "fixed",
        "events_per_bin": args.events_per_bin if args.count_bin else "",
        "time_res": args.time_res if not args.adaptive_bin and not args.count_bin else "",
        "positive_dist_thresh": args.positive_dist_threshold,
        "patch_or_frame": patch_or_frame,
        "seq_match_type": args.seq_match_type,
    }

    def is_equal(a, b):
        if pd.isna(a) or a == '':
            return pd.isna(b) or b == ''
        if pd.isna(b) or b == '':
            return False
        return str(a) == str(b)

    mask = pd.Series(True, index=csv_df.index)
    for k, v in row_signature.items():
        mask &= csv_df[k].apply(lambda x: is_equal(x, v))

    if mask.sum() == 0:
        print("\n❌ No matching row found. Debug info:")
        for k, v in row_signature.items():
            unique_vals = csv_df[k].unique()
            print(f" - {k:20}: target={v!r}, in_csv={unique_vals}")


    if mask.sum() == 1:
        idx = csv_df[mask].index[0]
        auc_val = csv_df.loc[idx, 'auc']
        
        # ✅ Skip if AUC is already populated
        if not (pd.isna(auc_val) or auc_val == ''):
            print(f"⏩ Skipping {ref_seq} vs {qry_seq}: AUC already filled ({auc_val})")
            return
        
        try:
            S, sim_time = compute_similarity_matrices(simMatPath, args, metric)
        except Exception as e:
            print(f"Failed to load simmat: {e}")
            return

        test_ds = TestDataset(args.database_folder, args.queries_folder,
                            positive_dist_threshold=args.positive_dist_threshold,
                            image_size=args.image_size, use_labels=args.use_labels)
        positives_per_query = test_ds.get_positives()
        start_match = time.time()

        if patch_or_frame == 'patch':
            num_queries, num_refs = S[0].shape
            combined_smats = np.zeros((num_queries, num_refs), dtype=np.float32)
            for j in range(args.num_patches):
                seqMat_patch, _ = plot_simmat_with_matches(S[j], positives_per_query, None,
                                                        patch_num=f'_Patch{j}-{args.num_patches}',
                                                        seq_len=seq_len, seq_match_type=args.seq_match_type)
                combined_smats += np.array(seqMat_patch)
            _, all_matches = plot_simmat_with_matches(combined_smats, positives_per_query, None,
                                                    patch_num='patch_combined', seq_len=1, seq_match_type=args.seq_match_type)
            _, auc, _ = getPRCurve_sim(combined_smats, positives_per_query)
        else:
            seqMat, all_matches = plot_simmat_with_matches(S, positives_per_query, None,
                                                        patch_num='frame', seq_len=seq_len, seq_match_type=args.seq_match_type)
            _, auc, _ = getPRCurve_sim(np.array(seqMat), positives_per_query)

        correct = sum(best_match_idx in positives_per_query[query_idx]
                    for query_idx, best_match_idx, _, _ in all_matches)
        recall_at_1 = correct / len(all_matches) if len(all_matches) > 0 else 0.0

        # Proceed with update if AUC is missing
        runtime = time.time() - start_match

        # Update full row with missing or default values
        updated_row = {
            'recall_at_1': f"{recall_at_1:.4f}",
            'auc': f"{auc:.4f}",
            'runtime': f"{runtime:.2f}",
            'patch_rows': args.patch_num_rows,
            'patch_cols': args.patch_num_cols,
        }

        for col, val in updated_row.items():
            if col in csv_df.columns:
                csv_df.loc[idx, col] = val
            else:
                print(f"⚠ Column {col} missing from CSV schema!")

        csv_df.to_csv(csv_path, index=False)
        print(f"✔ Updated row in {csv_path.name} for {ref_seq} vs {qry_seq}")


    else:
        print(f"⚠ Could not uniquely match row for {ref_seq} vs {qry_seq} ({mask.sum()} matches)")


if __name__ == "__main__":

    save_results = True
    saveSimMat = True

    args = parse.parse_arguments()

    run_vpr_save_results(args)







