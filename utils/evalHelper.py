import numpy as np
import cv2
np.random.seed(1)
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import imageio.v2 as imageio
from pathlib import Path

'''-------------------------------stat measure for picking a patch------------------------------------'''

def patch_certainity(frame, rows, cols):
    # frame=np.tanh(frame)
    # frame=random_downsample(frame,20,100)
    num_patches= rows*cols
    # Frame size
    height, width = frame.shape

    # Calculate the size of each patch
    patch_h = height // rows
    patch_w = width // cols

    
    # List to store the entropy of each patch
    entropies = []
    
    # Loop through each patch and compute the entropy
    for j in range(num_patches):
        row, col = divmod(j, cols)
        # Define the patch coordinates

        y_start, y_end = row * patch_h, (row + 1) * patch_h
        x_start, x_end = col * patch_w, (col + 1) * patch_w
                
        patch = frame[y_start:y_end, x_start:x_end]
        
        # Calculate entropy of the patch
        entropies.append(shannon_entropy(patch))
    
    # Find the patches with the lowest entropy
    # entropies=
    # sorted_indices = np.argsort(entropies)#[::-1]
    

    return np.array(entropies)


def min_max_normalize(data, new_min=0, new_max=1):
    if data.size == 0:
        return np.array([])  # Return empty array if input is empty

    min_val, max_val = np.min(data), np.max(data)
    if min_val == max_val:
        return np.full_like(data, new_min)  # Avoid division by zero

    return new_min + (data - min_val) * (new_max - new_min) / (max_val - min_val)


def shannon_entropy(image):
    """Calculate Shannon entropy of an image or distribution"""
    import numpy as np
    
    if len(image.shape) == 2:  # If it's an image
        # Calculate histogram
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)
    else:  # If it's already a distribution
        hist = image
        
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    
    # Calculate entropy
    if len(hist) > 0:
        return float(-np.sum(hist * np.log2(hist)))
    else:
        return 0.0
    

def calculate_match_quality(patch, match_idx, match_correlation_all, distances):
    metrics ={}
    '''--------------correlation based metrics----------------'''
    z_score = (match_correlation_all[match_idx] - np.mean(match_correlation_all)) /(np.std(match_correlation_all) if np.std(match_correlation_all) > 0 else 1)
    peak2mean = np.max(match_correlation_all) / np.mean(match_correlation_all) if np.mean(match_correlation_all) > 0 else 0
    metrics["corr_max"] = np.max(match_correlation_all)
    metrics["corr_mean"] = np.mean(match_correlation_all)
    metrics["corr_std"] = np.std(match_correlation_all)
    metrics["z_score"] = z_score
    metrics["peak_to_mean_ratio"] = peak2mean
    metrics["best_second_diff"] = np.sort(match_correlation_all)[-1] - np.sort(match_correlation_all)[-2]

    '''----------------patch content metrics--------------------'''
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

    max_val = np.max(patch)
    if max_val > 0:
        patch_normalized = (patch / max_val * 255).astype(np.uint8)
    else:
        # Handle the case where max_val is zero to avoid division by zero
        patch_normalized = np.zeros_like(patch, dtype=np.uint8)
    event_entropy = shannon_entropy(patch_normalized)
    random_noise = np.random.randn(*patch.shape)  
    neg_count = np.sum(patch<0)
    
    metrics["patch_sum"] = np.sum(abs(patch)) 
    metrics["patch_mean"] =np.mean(patch)
    metrics["patch_contrast"] =  np.std(patch)
    metrics["patch_entropy"] = event_entropy
    metrics["patch_polarity_ratio"] = np.sum(patch > 0) / (neg_count if neg_count > 0 else 1)
    
        
    '''----------------patch edge and corner metrics--------------------'''
    # Simple Sobel edge detection
    sobel_x = cv2.Sobel(patch_normalized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(patch_normalized, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    if np.sum(magnitude > 0) > 0:
        direction = np.arctan2(sobel_y, sobel_x)
        direction_hist = np.histogram(direction, bins=8, range=(-np.pi, np.pi))[0]
        direction_hist = direction_hist / np.sum(direction_hist)
        directional_bias = np.max(direction_hist) - np.min(direction_hist) # Calculate directional bias (how aligned are the gradients?)
    else:
        directional_bias = 0
    # FAST corner detection
    fast = cv2.FastFeatureDetector_create(threshold=10)
    keypoints = fast.detect(patch_normalized, None)
    # Harris corner response
    harris_response = cv2.cornerHarris(patch_normalized.astype(np.float32), 2, 3, 0.04)

    metrics["sobel_edge_density"] = np.sum(magnitude > 10) / patch.size
    metrics["sobel_grad_magnitude_mean"] = np.mean(magnitude)
    metrics["sobel_directional_bias"] = directional_bias
    metrics['fast_keypoint_count'] = len(keypoints)
    metrics['harris_response_mean'] = np.mean(harris_response)
    metrics['harris_response_max'] = np.max(harris_response)
    metrics["patch_noise_correlation"] = float(signal.fftconvolve(patch, random_noise, mode='valid')[0])
 
    metrics["match_distance"] = distances[match_idx]
    return metrics


'''-------------------------------Functions used for place rec------------------------------------'''

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth specified in decimal degrees (lat, lon).
    Returns distance in meters.
    """
    R = 6371000  # Radius of Earth in meters

    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c



def plot_ref_qry_with_overlap(ref_positions, qry_positions, threshold=25, figsize=(12, 10)):
    """
    Plot reference and query positions, highlighting areas where they overlap.
    
    Args:
        ref_positions: Array of reference positions (N x 2)
        qry_positions: Array of query positions (M x 2)
        threshold: Distance threshold in meters to consider positions as overlapping
        figsize: Size of the figure
    """
    # Convert to numpy arrays if they aren't already
    ref_positions = np.array(ref_positions)
    qry_positions = np.array(qry_positions)
    
    # Calculate pairwise distances between ref and qry positions
    distances = cdist(ref_positions, qry_positions)
    
    # Find minimum distance for each reference point to any query point
    min_distances = np.min(distances, axis=1)
    
    # Create overlap mask (True where distance is below threshold)
    overlap_mask = min_distances < threshold
    
    # Do the same for query points
    min_distances_qry = np.min(distances, axis=0)
    overlap_mask_qry = min_distances_qry < threshold
    print(sum(overlap_mask))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot reference trajectory
    ax.plot(ref_positions[:, 0], ref_positions[:, 1], 'b.', alpha=0.5, label='Reference Path')
    
    # Plot query trajectory
    ax.plot(qry_positions[:, 0], qry_positions[:, 1], 'g.', alpha=0.5, label='Query Path')
    
    # Highlight overlapping regions on reference path
    # ax.scatter(ref_positions[overlap_mask, 0], ref_positions[overlap_mask, 1], 
            #   c='r', alpha=0.8, s=25, label='Overlapping Regions')
    
    # Highlight overlapping regions on query path
    ax.scatter(qry_positions[overlap_mask_qry, 0], qry_positions[overlap_mask_qry, 1], 
              c='r', alpha=0.8, s=25, label='Query Overlap')
    
    # Add start and end markers
    ax.plot(ref_positions[0, 0], ref_positions[0, 1], 'bs', markersize=10, label='Reference Start')
    # ax.plot(ref_positions[-1, 0], ref_positions[-1, 1], 'b*', markersize=10, label='Reference End')
    ax.plot(qry_positions[0, 0], qry_positions[0, 1], 'gs', markersize=10, label='Query Start')
    # ax.plot(qry_positions[-1, 0], qry_positions[-1, 1], 'g*', markersize=10, label='Query End')
    
    # Set labels and title
    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m)', fontsize=12)
    ax.set_title('Reference and Query Paths with Overlapping Regions', fontsize=14)
    
    # Add legend
    ax.legend(loc='best')
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Add stats
    overlap_percentage_ref = 100 * np.sum(overlap_mask) / len(overlap_mask)
    overlap_percentage_qry = 100 * np.sum(overlap_mask_qry) / len(overlap_mask_qry)
    
    stats_text = (f"Reference path: {len(ref_positions)} points, {overlap_percentage_ref:.1f}% overlap\n"
                 f"Query path: {len(qry_positions)} points, {overlap_percentage_qry:.1f}% overlap")
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    return fig, ax



def vizRefQuery(refFrames, queryFrames, fps=6, tag='Event'):
    output_video = f'{tag}_RefvQry_Viz.mp4'
    writer = imageio.get_writer(output_video, fps=fps, codec="libx264", quality=10, format='ffmpeg')


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    for i in range(min(len(refFrames), len(queryFrames))):
        ax1.clear()
        ax2.clear()

        # Convert frames based on tag
        
        if tag == 'RGB':
            left_frame = (refFrames[i])
            right_frame = (queryFrames[i])
        else:
            left_frame = (np.tanh(refFrames[i]))
            right_frame = (np.tanh(queryFrames[i]))

        # Display frames in Matplotlib
        ax1.imshow(left_frame)
        ax1.set_title('Reference Frame')
        ax1.axis('off')

        ax2.imshow(right_frame)
        ax2.set_title('Query Frame')
        ax2.axis('off')

        # plt.pause(0.01)  # Adjust pause time for visualization speed
        # plt.draw()

        # Convert figure to numpy array and save frame
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        writer.append_data(frame)

    writer.close()
    plt.close()
    print(f"Video saved as {output_video}")