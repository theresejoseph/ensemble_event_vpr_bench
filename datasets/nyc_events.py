import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import math 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
from scipy.spatial import cKDTree
import numpy as np
from itertools import combinations
import imageio
import datetime 
from utils.h5_utils import load_processed_sequence, save_processed_sequence


def angle_diff_deg(a, b):
    """
    Returns the signed difference (a–b) wrapped to [–180, +180).
    """
    d = a - b
    return (d + 180) % 360 - 180


def haversine_distance(latlon1, latlon2):
    """Compute haversine distance between arrays of lat/lon points (in meters)."""
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = np.radians(latlon1[:, 1]), np.radians(latlon1[:, 0])
    lat2, lon2 = np.radians(latlon2[:, 1]), np.radians(latlon2[:, 0])
    
    dlat = lat2[:, None] - lat1
    dlon = lon2[:, None] - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2[:, None]) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c  # shape (len(lat2), len(lat1))


def find_most_overlap_sequence_pair(all_positions, sequence_names, distance_threshold=5.0):
    """
    Args:
        all_positions: dict mapping sequence name -> Nx2 numpy array (longitude, latitude)
        distance_threshold: distance in meters to consider as "overlap"
    
    Returns:
        best_pair: tuple of (seq_name1, seq_name2)
        best_overlap: number of overlapping points
    """
    
    best_overlap = -1
    best_pair = None

    for seq1_name, seq2_name in combinations(sequence_names, 2):
        pos1 = all_positions[seq1_name]
        pos2 = all_positions[seq2_name]

        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(pos1)
        distances, _ = tree.query(pos2, distance_upper_bound=distance_threshold)

        # Count how many points are within distance threshold
        overlap_count = np.sum(distances < distance_threshold)
        
        if overlap_count > best_overlap:
            best_overlap = overlap_count
            best_pair = (seq1_name, seq2_name)
    
    return best_pair, best_overlap


class NYCEventDataset_Bench:
    def __init__(self, base_path, image_size=(1280, 720)):
        self.base_path = base_path
        self.image_size = image_size


    def check_processed_data(self, processed_path, time_res):
        pass


    def process_sequence(self, args, reforqry, reconstructor=None, ref_gt_positions=None):
        """
        Loads images and ground-truth positions from the specified folder.
        Optionally filters query images based on valid indices computed from the reference
        ground-truth positions. If filtering is applied, only the images corresponding to 
        the valid query indices are loaded.
        
        Returns:
            For reference sequences or query sequences without filtering:
                tuple: (images, gt_positions)
            For query sequences with filtering (i.e. ref_gt_positions is provided):
                If gt_pos_only==True:
                    tuple: (None, gt_positions, valid_indices, closest_point_data)
                Else:
                    tuple: (images, gt_positions, valid_indices, closest_point_data)
        """
        sequence_name = args.sequences[args.ref_seq_idx]
        if reforqry == 'ref':
            subfolder = "ref"
        elif reforqry == 'qry':
            subfolder = "query"
        else:
            raise ValueError("reforqry must be either 'ref' or 'qry'")
        
        # Build the folder path
        folder_path = os.path.join(args.dataset_path, sequence_name, subfolder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path not found: {folder_path}")
        files = sorted(os.listdir(folder_path))
        file_paths=[]
        gt_positions = np.load(os.path.join(args.dataset_path, sequence_name)+'/ground_truth_new.npy', allow_pickle=True)
        print(gt_positions)
        valid_indices, closest_point_data=[], []
        
        # If images need to be loaded, determine whether to apply filtering (for query images)
        if reforqry == 'qry':
            for i, gt in enumerate(gt_positions):
                if gt[1] != []:
                    valid_indices.append(i)
                    closest_point_data.append((gt[1], 0, 0, 0))
                    file_paths.append(os.path.join(folder_path, files[i]))
            
            # Load only images at the valid indices
            images = []
            for i in tqdm(valid_indices, desc=f"Loading {sequence_name}/{subfolder} images"):
                try:
                    img = np.array(Image.open(file_paths[i]).convert("L")).astype(np.float32)
                    img_scaled = (img / 127.5) - 1
                    images.append(np.array(img_scaled))
                except Exception as ex:
                    print(f"Warning: Could not load image {file_paths[i]}: {ex}")
            return np.array(images), gt_positions[valid_indices], valid_indices, closest_point_data
        else:
            # For reference images or query images without filtering
            images, ref_ids = []
            file_paths= [os.path.join(folder_path, file) for file in files[::100]]
            for fp in tqdm(file_paths, desc=f"Loading {sequence_name}/{subfolder} images"):
                try:
                    img = np.array(Image.open(fp).convert("L")).astype(np.float32)
                    img_scaled = (img / 127.5) - 1
                    images.append(np.array(img_scaled))
                except Exception as ex:
                    print(f"Warning: Could not load image {fp}: {ex}")
            return np.array(images), gt_positions


class NYCEventDataset_Eval:
    def __init__(self, base_path, image_size=(1280, 720)):
        """
        Args:
            base_path (str): Base directory for the NYC-Event-VPR_VG dataset.
            image_size (tuple): Desired size for the image (width, height).
        """
        self.base_path = base_path
        self.image_size = image_size

    def parse_filename(self, filename):
        """
        Parses filenames in the format:
            @UTM_east @UTM_north @UTM_zone_number @UTM_zone_letter @latitude @longitude @pano_id @tile_num @heading @pitch @roll @height @timestamp @note @.jpg

        Returns:
            tuple: (UTM_east: float, UTM_north: float, heading: float)
        """
        name, _ = os.path.splitext(filename)
        parts = name.split('@')

        if len(parts) < 15:
            raise ValueError(f"Unexpected file name format (too few parts): {filename}")

        try:
            UTM_east = float(parts[1])
            UTM_north = float(parts[2])
            heading = float(parts[9])  # Might not exist if filename is too short
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing UTM_east/north/heading from '{filename}': {e}")
        
        

        return UTM_east, UTM_north, heading

    def check_processed_data(self, processed_path, time_res):
        pass

    def greedyTSP(self, gt_positions):
        coords = np.array(gt_positions)
        if len(coords) == 0:
            return []

        remaining = set(range(len(coords)))
        current = 0
        tsp_order = [current]
        remaining.remove(current)

        while remaining:
            remaining_list = list(remaining)
            dists = np.linalg.norm(coords[remaining_list] - coords[current], axis=1)
            next_idx = remaining_list[np.argmin(dists)]
            tsp_order.append(next_idx)
            remaining.remove(next_idx)
            current = next_idx

        return tsp_order
    
    def ref_qry_positions(self, args, reforqry, min_dist_m: float = 1.0, lat_bounds: tuple[float, float] = (582000, 585000), lon_bounds: tuple[float, float] = (4495000,4505000),):
        sequence_name = args.sequences[args.ref_seq_idx]
        if reforqry == 'ref':
            subfolder = "database"
        elif reforqry == 'qry':
            subfolder = "queries"
        else:
            raise ValueError("reforqry must be either 'ref' or 'qry'")
        
        # Build the folder path
        folder_path = os.path.join(self.base_path, "NYC-Event-VPR_Event", "images", sequence_name, subfolder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Path not found: {folder_path}")
        
        gps_positions: list[tuple[float, float]] = []
        file_paths:     list[str]             = []
        yaws:           list[float]           = []

        for fn in sorted(os.listdir(folder_path)):
            if not fn.lower().endswith(".jpg"):
                continue

            try:
                lat, lon, yaw = self.parse_filename(fn)
            except Exception as ex:
                print(f"Warning: skipping {fn!r}: {ex}")
                continue

            # optional lat/lon bounding box
            if lat_bounds and not (lat_bounds[0] < lat < lat_bounds[1]):
                continue
            if lon_bounds and not (lon_bounds[0] < lon < lon_bounds[1]):
                continue

           
            gps_positions.append((lat, lon, yaw))
            file_paths.append(os.path.join(folder_path, fn))
            yaws.append(yaw)

        return np.array(gps_positions), file_paths, np.array(yaws)

    def process_sequence(self, args, reforqry, reconstructor=None, ref_gt_positions=None):
        """
        Loads images and ground-truth positions from the specified folder.
        Optionally filters query images based on valid indices computed from the reference
        ground-truth positions. If filtering is applied, only the images corresponding to 
        the valid query indices are loaded.
        """
        gt_positions, file_paths, yaws=self.ref_qry_positions(args, reforqry)
        
        # If images need to be loaded, determine whether to apply filtering (for query images)
        if reforqry == 'qry' and ref_gt_positions is not None:
            # Compute valid indices and closest point data for query images first
            tsp_order=self.greedyTSP(gt_positions)
            gt_positions = gt_positions[tsp_order] 
            file_paths = [file_paths[i] for i in tsp_order]
            valid_indices = []
            closest_point_data = []
            for i, gt in enumerate(gt_positions):
                closest_point_id, min_distance, dist_sorted_frame_ids, distances = find_closest_pos(ref_gt_positions, gt, latlong=args.latlong)
                # then in your code:
                yaw_ref = ref_gt_positions[closest_point_id, 2]
                yaw_diff = angle_diff_deg(yaws[i], yaw_ref)

                if (min_distance < args.min_dist_toler and abs(yaw_diff) < 30):
                    valid_indices.append(i)
                    closest_point_data.append((closest_point_id, min_distance, dist_sorted_frame_ids, distances))
            
            # Load only images at the valid indices
            images = []
            for i in tqdm(valid_indices, desc=f"Loading {reforqry} images"):
                try:
                    img = np.array(Image.open(file_paths[i]).convert("L")).astype(np.float32)
                    # img = np.tanh(np.array(img).astype(np.float32))
                    img_scaled = (img / 127.5) - 1
                    images.append(np.array(img_scaled))
                except Exception as ex:
                    print(f"Warning: Could not load image {file_paths[i]}: {ex}")
            return np.array(images), gt_positions[valid_indices], valid_indices, closest_point_data
        else:
            # For reference images or query images without filtering
            images = []
            for fp in tqdm(file_paths, desc=f"Loading {reforqry} images"):
                try:
                    img = np.array(Image.open(fp).convert("L")).astype(np.float32)
                    # img = np.tanh(np.array(img).astype(np.float32))
                    img_scaled = (img / 127.5) - 1
                    images.append(np.array(img_scaled))
                except Exception as ex:
                    print(f"Warning: Could not load image {fp}: {ex}")
            return np.array(images), gt_positions
        

class NYCEventDataset:
    def __init__(self, base_path, sensor_size=(1280, 720)):
        self.base_path = base_path
        self.sensor_size = sensor_size

    def check_processed_data(self, path, time_res, endIdx):
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            return (
                data["frame_duration"] == time_res and
                data["endIdx"] == endIdx
            )
        except Exception as e:
            print(f"Failed to load processed data: {e}")
            return False
        
    def load_events_raw(self, sequence_name):
        import metavision_sdk as mv
        raw_folder = os.path.join(self.base_path, sequence_name)
        raw_files = sorted(glob.glob(os.path.join(raw_folder, "*.raw")))
        # Initialize the reader
        reader = mv.data.RawFileReader(raw_files[0])
        
        # Prepare lists to store event data
        timestamps = []
        x_coords = []
        y_coords = []
        polarities = []

        # Loop through the events in the raw file
        for event in reader:
            timestamps.append(event.timestamp)
            x_coords.append(event.x)
            y_coords.append(event.y)
            polarities.append(event.polarity)

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        polarities = np.array(polarities)

        # Return events as structured numpy array (t, x, y, p)
        events = np.core.records.fromarrays([timestamps, x_coords, y_coords, polarities],
                                            names="t, x, y, p", 
                                            formats="f8, u2, u2, u1")
        return events

    def load_gps_csv(self, sequence_name):
        csv_folder = os.path.join(self.base_path, sequence_name)
        csv_file=glob.glob(os.path.join(csv_folder, "*.csv"))[0]
        df = pd.read_csv(csv_file)  # or ',' depending on actual separator
    
        lat_lon = df[['Latitude', 'Longitude', 'HeadMotion']].to_numpy()
        gps_strings = df['Timestamp']
        gps_ts_micro = []
        for gps_str in gps_strings:
            date_part, us_part = gps_str.rsplit('_', 1)
            dt = datetime.datetime.strptime(date_part, "%Y-%m-%d_%H-%M-%S")
            timestamp_in_us = int(dt.timestamp() * 1e6) + int(us_part)
            gps_ts_micro.append(timestamp_in_us)
        timestamps= np.array(gps_ts_micro, dtype=np.int64)
    
        return lat_lon, timestamps


    def get_slice_indices(self, ev_ts, gps_ts, time_res):
        print(ev_ts[:10], gps_ts[:10])
        end_indices = np.searchsorted(ev_ts, gps_ts)
        if time_res > 1:
            start_indices = [max(0, end - self.sensor_size[0] * self.sensor_size[1]) for end in end_indices]
        else:
            time_res=int(time_res*1e9)
            start_indices = np.searchsorted(ev_ts, gps_ts - time_res)
        return start_indices, end_indices

    

    def process_sequence(self,  args, reforqry, reconstructor, gt_pos_only=False):
        """
        Loads event data and GPS positions from the specified sequence folder.
        For queries, filters the frames using reference positions based on spatial
        and yaw proximity, if provided.
        """
        self.plot_gt_paths(args) 
        # id = args.qry_seq_idx if reforqry == 'qry' else args.ref_seq_idx
        seq_name = args.sequences[args.ref_seq_idx] if reforqry == 'ref' else args.sequences[args.qry_seq_idx]
        incr = args.ref_incr if reforqry == 'ref' else args.query_incr


        # Paths
        processed_path = f"processed_data/{seq_name}_t{args.time_res}_e{args.end_idx}.npz"

        # Check if we need to process
        if self.check_processed_data(processed_path, args.time_res, args.end_idx):
            print(f"Loading existing processed data for {seq_name}")
            event_frames, gt_positions = load_processed_sequence(processed_path, startIdx=args.start_idx, endIdx=args.end_idx, incr=incr, gt_pos_only=gt_pos_only)
        else:
            print(f"Reconstructing frames for {seq_name} at time_res={args.time_res}...")
            events = self.load_events_raw(seq_name)
            gt_positions, timestamps = self.load_gps_csv(seq_name)

            start_indices, end_indices = self.get_slice_indices(events['t'], timestamps, args.time_res)
            print(start_indices,end_indices)

            event_frames, frame_times = reconstructor.reconstruct(eventsData=events, sensor_size=self.sensor_size,
                start_indices=start_indices, end_indices=end_indices)

            save_processed_sequence(processed_path, event_frames, gt_positions, frame_times, None, args.time_res, args.end_idx)

        return event_frames, gt_positions


    def plot_gt_paths(self, args):
        # Generate a colormap with N unique colors
        num_seqs = len(args.sequences)
        colors = cm.get_cmap('tab20', num_seqs)  # 'tab10' or 'hsv' or any other

        plt.figure()
        all_positions={}
        for idx, sequence_name in enumerate(args.sequences):
            gt_positions, timestamps = self.load_gps_csv(sequence_name)
            all_positions[sequence_name]=gt_positions
            color = colors(idx)
            plt.plot(gt_positions[:, 0], gt_positions[:, 1], color=color, label=idx, alpha=0.7)

        plt.legend(loc='best')
        plt.title("Ground Truth Positions per Sequence")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig('./gt_pos_all.png')

        best_pair, best_overlap=find_most_overlap_sequence_pair(all_positions, args.sequences)
        print(best_pair, best_overlap)
        plt.clf()
        plt.plot(all_positions[best_pair[0]][:, 0], all_positions[best_pair[0]][:, 1])
        plt.plot(all_positions[best_pair[1]][:, 0], all_positions[best_pair[1]][:, 1])
        plt.savefig('./gt_pos.png')    

