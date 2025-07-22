import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# import imageio.v2 as imageio
# import cv2
# from matplotlib.gridspec import GridSpec
import seaborn as sns
# from matplotlib.colors import Normalize
# from scipy.stats import pearsonr, spearmanr
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy import signal
from pathlib import Path

class MatchVisualizer:
    def __init__(self, save_path="match_video.mp4", fps=10):
        """Initialize the visualizer and video settings"""
        self.fig, self.axes = plt.subplots(3,3, figsize=(12, 6))
        self.axes = self.axes.flatten()
        self.save_path = save_path
        self.fps = fps
        self.corr_frames = []  # Store data for each frame
        self.incorr_frames = []  # Store data for each frame      

    def update(self, query_frame, match_frame, query_pos, match_pos, dist, corrCounter, evalCounter, gtMatch, minDist, matchCorrelation, dMat, match_idx, closest_point_id, rank,distToler=10):
        """Store frame data for later rendering"""
        frame_data = (query_frame, match_frame, query_pos, match_pos, dist, corrCounter, 
                      evalCounter, gtMatch, minDist, matchCorrelation, dMat, match_idx, closest_point_id, rank,distToler)
        if dist < distToler:
            self.corr_frames.append(frame_data)
        elif dist > distToler:
            self.incorr_frames.append(frame_data)
        self._render_frame(frame_data)
        # plt.draw()
        # plt.pause(1)
        plt.savefig('./results/frameMatching.png')

    def _render_frame(self, frame_data):
        if hasattr(self, 'colorbars'):
            for cbar in self.colorbars:
                if cbar is not None:
                    cbar.remove()
        self.colorbars = []
        """Render a single frame"""
        query_frame, match_frame, query_pos, match_pos, dist, corrCounter, evalCounter, gtMatch, minDist, matchCorrelation, dMat, match_idx, closest_point_id, rank,distToler = frame_data
        color = 'g' if dist < distToler else 'r'
        alpha = 1.0 if dist < distToler else 0.1
        
        self.axes[0].clear(), self.axes[1].clear, self.axes[2].clear(), self.axes[8].clear()

        # Subplot 1: Query Frame

        im0=self.axes[0].imshow(query_frame)
        self.axes[0].set_title(f"Query {evalCounter}")
        cbar=self.fig.colorbar(im0, ax=self.axes[0]) 
        self.colorbars.append(cbar)

        # Subplot 2: Match Frame
        im1=self.axes[1].imshow(np.tanh(match_frame))#, cmap='bwr'
        self.axes[1].set_title(f"Match {match_idx}: {round(dist, 2)}m away", color=color)
        cbar=self.fig.colorbar(im1, ax=self.axes[1]) 
        self.colorbars.append(cbar)

        # Subplot 3: Closest match
        im2=self.axes[2].imshow(np.tanh(gtMatch))
        self.axes[2].set_title(f'GT_match {closest_point_id}: {round(minDist, 2)}m away', color=color)
        cbar=self.fig.colorbar(im2, ax=self.axes[2]) 
        self.colorbars.append(cbar)

        # Subplot 4 and 5: Positions
        if dist<distToler:
            self.axes[3].scatter(query_pos[0], query_pos[1], color='black', s=2, label='Query Position')
            self.axes[3].scatter(match_pos[0], match_pos[1], color=color, s=2, label='Match Position')
            self.axes[3].plot([query_pos[0], match_pos[0]], [query_pos[1], match_pos[1]], linestyle='-', alpha=alpha, color=color)
            self.axes[3].set_title("Correct Match Positions")
            
            self.axes[3].grid(True)
        else:
            self.axes[4].scatter(query_pos[0], query_pos[1], color='black', s=2, label='Query Position')
            self.axes[4].scatter(match_pos[0], match_pos[1], color=color, s=2, label='Match Position')
            self.axes[4].plot([query_pos[0], match_pos[0]], [query_pos[1], match_pos[1]], linestyle='-', alpha=alpha, color=color)
            self.axes[4].set_title("Incorrect Match Positions")
            # self.axes[4].set_ylabel("Y [m]")
            self.axes[4].grid(True)
            

        self.axes[5].plot(evalCounter,corrCounter/evalCounter,'.', color='b')
        self.axes[5].set_title(f'Recall: {corrCounter/evalCounter:.2f}')

        # self.axes[6].imshow(matchCorrelation, cmap='plasma')
        # self.axes[6].set_title('Match Correlation across shifts')

        self.axes[7].imshow(dMat, cmap='plasma')
        self.axes[7].set_title('Distance Matrix')

        self.axes[8].bar(np.arange(len(rank)), rank, color='b')
        self.axes[8].set_title('Correct Match Rank')
        
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
    
    def corr_animate(self, i):
        self._render_frame(self.corr_frames[i])

    def incorr_animate(self, i):
        self._render_frame(self.incorr_frames[i])
        
    def save_video(self):
        """Save all frames as a video using Matplotlib's animation functionality"""
        if self.corr_frames:
            # Create a new animation using the stored frame data
            ani = animation.FuncAnimation(self.fig, self.corr_animate, frames=len(self.corr_frames), repeat=False)
            writer = animation.FFMpegWriter(fps=self.fps, codec='libx264', bitrate=1800)
            ani.save('TP_'+self.save_path, writer=writer)
            print(f"✅ Video saved at TP {self.save_path}")
        else:
            print("No correct frames to save!")
        
        self.axes[5].clear()
        if self.incorr_frames:
            # Create a new animation using the stored frame data
            ani = animation.FuncAnimation(self.fig, self.incorr_animate, frames=len(self.incorr_frames), repeat=False)
            writer = animation.FFMpegWriter(fps=self.fps, codec='libx264', bitrate=1800)
            ani.save('FP_'+self.save_path, writer=writer)
            print(f"✅ Video saved at FP {self.save_path}")
        else:
            print("No incorrect frames to save!")
    

# Create a figure with a more organized layout
# fig = plt.figure(figsize=(16, 12))
# gs = GridSpec(4, 3, figure=fig)
def plot_patch_matching(params, query_frame, ref_frame, distToler=10):

    qry_patches = params["qry_patches"]
    ranks = params["ranks"]
    m_ids = params["m_ids"]
    ref_patches = params["ref_patches"]
    closest_patch_id = params["closest_patch_id"]
    predicted_patch_id = params["pred_best_patch_id"]
    matchedIds_closest= params["matchedIds_closest"]
    recall_history_closest= params["recall_history_closest"] 
    recall_history_predicted= params["recall_history_predicted"]
    dist_history_closest= params["dist_history_closest"]
    dist_history_predicted= params["dist_history_predicted"]
    matchedIds_pred= params["matchedIds_pred"]
    closest_dist=params["closest_dist"]      
    predicted_dist=params["predicted_dist"]

    plt.clf()
    # Extract patches for visualization
    closest_qry_patch = qry_patches[closest_patch_id]
    predicted_qry_patch = qry_patches[predicted_patch_id]
    closest_ref_patch = ref_patches[0]  # Closest reference patch
    predicted_ref_patch = ref_patches[1]  # Predicted reference patch
    closest_corr = ranks[closest_patch_id]
    predicted_corr = ranks[predicted_patch_id]

    # Row 1: Query patches and full frame
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.tanh(closest_qry_patch), cmap='viridis')
    ax1.set_title(f"Closest Query Patch (ID: {closest_patch_id})")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.tanh(predicted_qry_patch), cmap='viridis')
    ax2.set_title(f"Predicted Best Query Patch (ID: {predicted_patch_id})")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.tanh(query_frame), cmap='viridis')
    ax3.set_title("Full Query Frame")
    ax3.axis('off')
    
    # Row 2: Reference patches and full frame
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(np.tanh(closest_ref_patch), cmap='viridis')
    ax4.set_title(f"Closest Ref Patch (ID: {m_ids[closest_patch_id]})\nCorr: {closest_corr[m_ids[closest_patch_id]]:.3f} Dist: {closest_dist:.2f}")
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(np.tanh(predicted_ref_patch), cmap='viridis')
    ax5.set_title(f"Predicted Ref Patch (ID: {m_ids[predicted_patch_id]})\nCorr: {predicted_corr[m_ids[predicted_patch_id]]:.3f} Dist: {predicted_dist:.2f}")
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(np.tanh(ref_frame), cmap='viridis')
    ax6.set_title("Ground Truth Reference Frame")
    ax6.axis('off')
    

    # Row 3: Metrics
    # Histogram for correlation values - Fix: plot histograms separately
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_title("Similarity Matrix")
    ax7.imshow(params["simMat"])
    
    # Histogram for distance values - Fix: plot histograms separately
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(dist_history_closest,'.-', color='blue', label='Closest', alpha=0.7)
    ax8.plot(dist_history_predicted, '.-', color='orange', label='Predicted', alpha=0.7)
    ax8.axhline(y=distToler, color='r', linestyle='--', label=f'Tolerance: {distToler}')
    ax8.set_title("Distance Values")
    ax8.set_xlabel("Query Instance")
    ax8.set_ylabel("Distance")
    ax8.legend()
    
    
    # Line plot for recall over query instances
    ax9 = fig.add_subplot(gs[2, 2])
    x_vals = range(1, len(recall_history_closest) + 1)
    ax9.plot(x_vals, recall_history_closest, 'b-', label='Closest Recall')
    ax9.plot(x_vals, recall_history_predicted, 'orange', label='Predicted Recall')
    ax9.set_title(f"Recall pred:{recall_history_predicted[-1]}, close:{recall_history_closest[-1]}")
    ax9.set_xlabel("Query Instance")
    ax9.set_ylabel("Recall")
    ax9.set_ylim([0, 1.1])
    ax9.grid(True, linestyle='--', alpha=0.7)
    ax9.legend()

    # Histogram for distance values - Fix: plot histograms separately
    ax10 = fig.add_subplot(gs[3, 1])
    ax10.plot(matchedIds_closest,'.-', color='blue', label='Closest', alpha=0.7)
    ax10.plot(matchedIds_pred, '.-', color='orange', label='Predicted', alpha=0.7)
    ax10.set_title("Match_Index")
    ax10.set_xlabel("Query Instance")
    ax10.set_ylabel("Distance")
    ax10.legend()

    # ax11 = fig.add_subplot(gs[3, 0])
    # ax11.set_title("Sequence Similarity Matrix")
    # ax11.imshow(params["seqMat"])
    
    
    
    plt.tight_layout()
    # Capture the frame
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    return frame


def compute_pr_data(simMat, valid_indices, closest_point_data, args, sequence_lens):
    """
    Computes precision-recall (PR) data for different sequence lengths.
    
    Args:
        simMat: 2D numpy array of similarity values (each row corresponds to a query frame).
        valid_indices: List/array of valid query indices.
        closest_point_data: List of tuples per query: (closest_point_id, min_distance, dist_sorted_frame_ids, distances).
        args: Object with experiment parameters (e.g. args.dist_thresh).
        sequence_lens: List of sequence lengths to evaluate.
    
    Returns:
        results: Dictionary mapping each sequence length to a dict containing:
            - 'precisions': List of precision values (accumulated).
            - 'recalls': List of recall values (accumulated).
            - 'auc': Area under the PR curve.
            - 'seqMat': List of sequence matching outputs (for each query).
            - 'matchIds': List of match ID tuples.
            - 'all_matches': List of tuples (query_idx, best_match_id, match_similarity, distance).
    """
    results = {}

    for sequence_len in sequence_lens:
        seqMat = []
        all_matches = []
        matchIds = []
        match_results=[]
        # Ensure simMat is a NumPy array
        simMat = np.array(simMat)
        for i in range(len(simMat)):
            query_idx = valid_indices[i]
            # Get the distances vector from closest_point_data
            _, _, _, distances = closest_point_data[i]
            
            if sequence_len == 1 or i < sequence_len:
                current_sim = simMat[i]
                best_match_id = np.argmax(current_sim)
                match_similarity = np.max(current_sim)
                seqMat.append(current_sim)
            else:
                simForNorm=simMat.copy()
                # normsim=  (simForNorm[:i] - simForNorm[:i].min()) / (simForNorm[:i].max() - simForNorm[:i].min() + 1e-8)
                # curr_simMat = normsim[max(0, i - sequence_len):]
                curr_simMat = simMat[max(0, i - sequence_len):i]
                curr_simMat = preprocessing.normalize(curr_simMat, axis=1)#(curr_simMat - curr_simMat.min()) / (curr_simMat.max() - curr_simMat.min() + 1e-8)
                pad_width = sequence_len - 1
                curr_simMat = np.pad(curr_simMat, ((0, 0), (pad_width, 0)), mode='constant', constant_values=0)
                diag_kernel = np.eye(sequence_len)
                
                seqMatchOut = signal.fftconvolve(curr_simMat, diag_kernel, mode='valid').flatten()
                best_match_id = np.argmax(seqMatchOut)
                seqMat.append(seqMatchOut)
                match_similarity = np.max(seqMatchOut)
                
            matchIds.append((i, best_match_id))
            all_matches.append((query_idx, best_match_id, match_similarity, distances[best_match_id]))
            match_results.append((match_similarity, int((distances[best_match_id] < args.dist_thresh))))


        all_matches.sort(key=lambda x: x[2], reverse=True)

        total_matches = len(all_matches)
        total_relevant = sum(1 for (query_idx, best_match_id, match_similarity, dist) in all_matches if dist < args.dist_thresh)
       
        precisions = []
        recalls = []
        tp_count, fp_count = 0, 0
        for (_, _, similarity, distance) in all_matches:
            if distance < args.dist_thresh:
                tp_count += 1
            else:
                fp_count += 1
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
            recall = tp_count / total_relevant if total_relevant > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)

        pr_auc = np.trapz(precisions, recalls) if recalls and precisions else 0.0
        

        # # now build precision/recall via sklearn
        # sims, relevances = zip(*match_results)
        # precisions, recalls, _ = precision_recall_curve(relevances, sims)
        # pr_auc = auc(recalls, precisions)

        results[sequence_len] = {
            'precisions': precisions,
            'recalls': recalls,
            'auc': pr_auc,
            'seqMat': seqMat,
            'matchIds': matchIds,
            'all_matches': all_matches,
        }
    return results


def plot_simmat_with_matches(seqMat, all_matches, args, save_path):

    plt.figure(figsize=(12, 8))
    # Display the similarity matrix with an enhanced colormap (e.g., 'viridis' or 'hot')
    plt.imshow(seqMat, cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(label='Similarity Score')
    plt.title('Similarity Matrix with Matches')
    # Plot incorrect matches (red dots)
    for i, info in enumerate(all_matches):
        query_idx, best_match_id, match_similarity, distance=info
        if distance > args.dist_thresh:
            plt.plot(best_match_id,query_idx,'o', color='red', markersize=1)
        else:
            plt.plot(best_match_id,query_idx,'x', color='lime', markersize=2)

    # Save the figure at high resolution (dpi=300) with tight bounding box
    save_path = Path(save_path)  # make sure it's a Path object
    new_path = save_path.with_stem(save_path.stem)
    plt.savefig(new_path.with_suffix('.png'), dpi=500, bbox_inches='tight')
    plt.close()


def plot_pr_curves_combined(results_standard, sequence_lens, output_path, title="Precision-Recall Curves"):
    """
    Plots PR curves for both patch and no-patch experiments on the same figure.
    
    Args:
        results_patch: Dictionary of PR data for the patch experiment, keyed by sequence length.
        results_standard: Dictionary of PR data for the standard (no patch) experiment.
        sequence_lens: List of sequence lengths used in evaluation.
        output_path: Path to save the plot image.
        title: Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Generate a pastel color palette
    pastel_colors = sns.color_palette("pastel", 7)

    # Generate a standard (more saturated) color palette
    standard_colors = sns.color_palette("deep", 7)
    for i, seq_len in enumerate(sequence_lens):
        color1 = pastel_colors[i % len(pastel_colors)]
        color2 = standard_colors[i % len(standard_colors)]
        standard_data = results_standard[seq_len]
        print(f"FRAME - Sequence Length {seq_len} - Prec@1 {standard_data['precisions'][-1]} - AUC {standard_data['auc']} ")

        # Standard curve: dashed line, triangle marker, slightly lighter color
        plt.plot(
            standard_data['recalls'],
            standard_data['precisions'],
            linestyle='-',
            marker='.',
            color=color2,
            linewidth=1.5,
            alpha=1,
            label=f'Standard SeqLen {seq_len} (P@1={standard_data["precisions"][-1]*100:.1f})'
        )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.savefig(output_path)
    plt.close()


def evaluate_similarity_matrix(simMatPath, valid_indices, closest_point_data, args, sequence_lens=[1, 6, 10, 20, 50, 75], output_path="combined_PR_curve.png"):
    """
    Loads a dictionary of similarity matrices (with keys 'patch' and 'standard') from file,
    computes PR data for each experiment over several sequence lengths, and
    plots both PR curves on the same figure.
    
    Args:
        simMatPath: Path to the .npy file containing a dict with keys "patch" and "standard".
        valid_indices: List/array of valid query indices.
        closest_point_data: List/array with closest point data for each query frame.
        args: Experiment configuration (should include args.dist_thresh, etc.).
        sequence_lens: List of sequence lengths to evaluate.
        output_path: File path where the combined PR plot will be saved.
    
    Returns:
        results: Dictionary with keys 'patch' and 'standard', each mapping to their PR evaluation results.
    """
    # Load the similarity matrices dictionary. Note: allow_pickle=True is required.
    simMat_all = np.load(simMatPath, allow_pickle=True).item()
    
    # Normalize each similarity matrix (using absolute values, as per your code)
    simMat_standard = simMat_all["standard"]
    # simMat_standard = preprocessing.normalize(simMat_standard)
    
    # Compute PR data for both patch and standard experiments.
    results_standard = compute_pr_data(simMat_standard, valid_indices, closest_point_data, args, sequence_lens)
    
    # Plot combined PR curves on a single figure.
    plot_pr_curves_combined(results_standard, sequence_lens, output_path, 
                            title=f"Precision-Recall Curves Patch vs Standard - {args.sequences[args.ref_seq_idx]}_vs_{args.sequences[args.qry_seq_idx]}_{args.end_idx}")
    
    for seq_len in sequence_lens:
        standard_data=results_standard[seq_len]
        sim_dir = os.path.dirname(simMatPath)
        simmat_save_path = os.path.join(sim_dir, f'{args.sequences[args.ref_seq_idx]}_vs_{args.sequences[args.qry_seq_idx]}_{args.end_idx}_seqLen{seq_len}_simMat.png')
        plot_simmat_with_matches(standard_data['seqMat'][seq_len:],  standard_data['all_matches'],  args, simmat_save_path)

    return {"standard": results_standard}


def compute_framewise_matches(simMat, valid_indices, closest_point_data, sequence_len, args):
    """
    Computes frame-by-frame matching outputs (without reordering) for a given similarity matrix.
    
    Args:
        simMat (np.ndarray): Normalized similarity matrix (each row for a query frame).
        valid_indices (list or np.ndarray): List of valid query indices.
        closest_point_data (list): List of tuples per query:
            (closest_point_id, min_distance, dist_sorted_frame_ids, distances).
        sequence_len (int): Sequence length used for matching.
        args: Object containing experiment parameters (e.g. args.dist_thresh).
    
    Returns:
        matches: List (length equal to number of frames) where each element is a tuple:
            (best_match_id, match_conf, distance)
    """
    num_frames = len(simMat)
    matches = []
    
    for i in range(num_frames):
        # Get the distance vector for this frame
        _, _, _, distances = closest_point_data[i]
    
        if sequence_len == 1 or i < sequence_len:
            current_sim = simMat[i]
            best_match_id = np.argmax(current_sim)
            match_conf = np.max(current_sim)
        else:
            # Compute sequence match using the last `sequence_len` frames
            curr_seq = simMat[max(0, i - sequence_len):i]
            curr_seq = preprocessing.normalize(curr_seq, axis=0)
            pad_width = sequence_len - 1
            curr_seq = np.pad(curr_seq, ((0,0), (pad_width, 0)), mode='constant', constant_values=0)
            diag_kernel = np.ones(sequence_len)
            seqMatchOut = signal.fftconvolve(curr_seq, np.diag(diag_kernel), mode='valid').flatten()
            best_match_id = np.argmax(seqMatchOut)
            match_conf = np.max(seqMatchOut)
        
        matches.append((best_match_id, match_conf, distances[best_match_id]))
    return matches


def create_match_video(qry_images, ref_images, valid_indices, closest_point_data, simMatPath, sequence_len,
                       ref_positions, qry_positions, args, output_path):
    """
    Generates and saves a video visualizing matching results in a 3-row layout.
    
    Layout:
      Top row:
         Left:   Query image (qry_images[frame_idx])
         Right:  Reference image using closest point match (from closest_point_data)
      
      Middle row:
         Left:   Patch-based match reference image (from patch_matches)
         Right:  Standard match reference image (from std_matches)
      
      Bottom row:
         Left:   Recall-over-time plot (for patch and standard)
         Right:  Map showing ref and qry positions with markers.
    
    Args:
        qry_images (List[np.ndarray]): Query images.
        ref_images (List[np.ndarray]): Reference images.
        valid_indices: List/array of valid query indices.
        closest_point_data: List of tuples per query frame.
        simMatPath (str): Path to .npy file containing dict {"patch": ..., "standard": ...}.
        sequence_len (int): Sequence length used for matching.
        ref_positions (np.ndarray): Reference positions (num_ref x 2).
        qry_positions (np.ndarray): Query positions (num_qry x 2).
        args: Contains at least args.dist_thresh and args.results_loc.
        output_path (str): Path for saving the video.
    """
    # --- Load similarity matrices ---
    simMat_all = np.load(simMatPath, allow_pickle=True).item()
    raw_standard = simMat_all["standard"]
    simMat_standard = preprocessing.normalize(np.abs(raw_standard))
    
    num_frames = len(simMat_standard)
    
    # --- Compute match outputs only once if not provided ---
    # patch_matches = compute_framewise_matches(simMat_patch, valid_indices, closest_point_data, sequence_len, args)
    std_matches = compute_framewise_matches(simMat_standard, valid_indices, closest_point_data, sequence_len, args)
    
    # --- Compute recall over time if not provided ---
    std_recalls = []
    std_tp = 0
    for i in range(num_frames):
        # distances from the precomputed matches
        _, _, std_dist = std_matches[i]
        if std_dist < args.dist_thresh:
            std_tp += 1
        std_recalls.append(std_tp / (i+1))
    
    # --- Set up figure layout (3 rows x 2 columns) ---
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])
    
    # Top row: query and closest match images
    ax_qry = fig.add_subplot(gs[0, 0])
    ax_closest = fig.add_subplot(gs[0, 1])
    # Middle row: patch match and standard match images
    ax_recall = fig.add_subplot(gs[1, 0])
    ax_std = fig.add_subplot(gs[1, 1])
    # Bottom row: recall plot and map
    ax_map = fig.add_subplot(gs[2, :])
    
    # Initialize axes for the images:
    img_qry = ax_qry.imshow(qry_images[0])
    ax_qry.set_title('Query Image')
    ax_qry.axis('off')
    
    # Closest point from closest_point_data (top row right)
    closest_id = closest_point_data[0][0]
    img_closest = ax_closest.imshow(ref_images[closest_id])
    ax_closest.set_title(f'Closest (ID: {closest_id})')
    ax_closest.axis('off')
    

    sm_idx, _, std_dist = std_matches[0]
    img_std = ax_std.imshow(ref_images[sm_idx])
    ax_std.set_title(f' Match (ID: {sm_idx}, Dist: {std_dist:.2f})')

    # Bottom left: Recall over time plot.
    line_std, = ax_recall.plot([], [], 'r.-', label='Standard Recall')
    ax_recall.set_xlim(0, num_frames)
    ax_recall.set_ylim(0, 1.05)
    ax_recall.set_xlabel('Frame Index')
    ax_recall.set_ylabel('Recall')
    ax_recall.set_title('Recall over Time')
    ax_recall.legend(loc='upper left')
    
    # Bottom right: Map plotting (full trajectories + current positions)
    ax_map.plot(ref_positions[:, 0], ref_positions[:, 1], '.', color='gray', label='Ref Path')
    ax_map.plot(qry_positions[:, 0], qry_positions[:, 1], '.', color='green', label='Qry Path')
    q_marker, = ax_map.plot([], [], 'bo', markersize=8, label='Current Query')
    std_marker, = ax_map.plot([], [], 'mo', markersize=8, label='Match')
    ax_map.set_title('Map: Query & Ref Positions')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    ax_map.legend(loc='upper right')
    ax_map.axis('equal')
    
    def update(frame_idx):
        # --- Top row ---
        # Update query image.
        img_qry.set_data(qry_images[frame_idx])
        ax_qry.set_title(f'Query Frame {frame_idx}')
        # Update closest match image (from closest_point_data)
        cp_id = closest_point_data[frame_idx][0]
        ax_closest.images[0].set_data(ref_images[cp_id])
        ax_closest.set_title(f'Closest (ID: {cp_id}, Dist: {closest_point_data[frame_idx][1]:.2f})')
        
        # --- Middle row ---
        sm_idx, sm_conf, sm_dist = std_matches[frame_idx]
        ax_std.images[0].set_data(ref_images[sm_idx])
        ax_std.set_title(f'Standard Match (ID: {sm_idx}, Dist: {sm_dist:.2f})')
        
        # --- Bottom left: Recall plot ---
        x_vals = list(range(frame_idx+1))
        line_std.set_data(x_vals, std_recalls[:frame_idx+1])
        ax_recall.relim()
        ax_recall.autoscale_view()
        
        # --- Bottom right: Map ---
        # Current query position.
        q_x, q_y = qry_positions[frame_idx][0],qry_positions[frame_idx][1]
        q_marker.set_data([q_x], [q_y])

        s_x, s_y = ref_positions[sm_idx][0],ref_positions[sm_idx][1]
        std_marker.set_data([s_x], [s_y])
        ax_map.set_title(f'Map: Frame {frame_idx} | Match Dist: {sm_dist:.2f}')
        
        return (img_qry, ax_closest.images[0], ax_std.images[0],line_std, q_marker, std_marker)
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=500, blit=False)
    ani.save(output_path, writer='ffmpeg', fps=4)
    print(f"✅ Saved video to {output_path}")
