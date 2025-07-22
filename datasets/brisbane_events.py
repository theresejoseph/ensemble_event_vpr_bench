# from tonic.datasets import VPR as TonicVPR
import os
import numpy as np
import pyproj
import pynmea2
from tqdm import tqdm
import sys
from scipy.interpolate import interp1d
from datasets.base_dataset import BaseDataset
from utils.odometry_utils import adaptive_bin_size, plot_vels_accs_bins


def convert_latlon_to_utm(positions):
    """
    Convert a list of (lat, lon) pairs in WGS84 to UTM (easting, northing) in meters,
    using the conventional false-northing for southern hemisphere.

    Returns:
      List of [easting, northing] in meters
    """
    # WGS84 geographic CRS
    wgs84 = pyproj.CRS("EPSG:4326")

    # pick first point to choose zone & hemisphere
    lat0, lon0 = positions[0]

    zone = int((lon0 + 180) / 6) + 1
    is_north = (lat0 >= 0)

    # choose EPSG code
    epsg_code = 32600 + zone if is_north else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)

    # build transformer
    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)

    # transform all points
    utm_positions = []
    for lat, lon in positions:
        easting, northing = transformer.transform(lon, lat)
        utm_positions.append([easting, northing])

    return np.array(utm_positions)


def interpolate_positions_at_intervals(xy_positions, metric_data_spacing):
    """
    Interpolate GPS positions to get positions at regular metric intervals.
    
    Args:
        xy_positions: Nx3 array with [x, y, timestamp] in meters
        metric_data_spacing: desired spacing between points in meters
        
    Returns:
        dict with keys:
            - 'positions': Mx3 array [x, y, timestamp] at regular intervals
            - 'distances': cumulative distances for each interpolated point
            - 'total_distance': total path distance
    """
    if len(xy_positions) < 2:
        return {'positions': xy_positions, 'distances': [0], 'total_distance': 0}
    
    # Calculate cumulative distances along the path
    cumulative_distances = [0]
    for i in range(1, len(xy_positions)):
        dist = np.sqrt((xy_positions[i, 0] - xy_positions[i-1, 0])**2 + 
                    (xy_positions[i, 1] - xy_positions[i-1, 1])**2)
        cumulative_distances.append(cumulative_distances[-1] + dist)
    
    cumulative_distances = np.array(cumulative_distances)
    total_distance = cumulative_distances[-1]
    
    # Create interpolation functions for x, y, and timestamp
    interp_x = interp1d(cumulative_distances, xy_positions[:, 0], 
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_y = interp1d(cumulative_distances, xy_positions[:, 1], 
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_t = interp1d(cumulative_distances, xy_positions[:, 2], 
                    kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Generate target distances at regular intervals
    num_points = int(total_distance / metric_data_spacing) + 1
    target_distances = np.arange(0, num_points * metric_data_spacing, metric_data_spacing)
    
    # Only keep distances within the actual path length
    target_distances = target_distances[target_distances <= total_distance]
    
    # Interpolate positions and timestamps at target distances
    interpolated_x = interp_x(target_distances)
    interpolated_y = interp_y(target_distances)
    interpolated_t = interp_t(target_distances)
    
    # Combine into final array
    interpolated_positions = np.column_stack((interpolated_x, interpolated_y))

    print(f"Original GPS points: {len(xy_positions)}")
    print(f"Interpolated points: {len(interpolated_positions)}")
    print(f"Total distance: {total_distance:.2f} meters")
    print(f"Spacing: {metric_data_spacing} meters")
    
    return interpolated_t, interpolated_positions


def find_first_self_loop(gt_positions: np.ndarray, threshold=10.0, min_gap=20):
    """
    Detects the first loop closure in a full trajectory based on proximity.
    
    Args:
        gt_positions: (N, 2) or (N, 3) numpy array of positions
        threshold: distance to count as a loop closure
        min_gap: number of indices to skip before considering it a valid match (avoids trivial neighbors)

    Returns:
        loop_idx_1: earlier index i
        loop_idx_2: later index j (j > i)
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(gt_positions)

    for j in range(min_gap, len(gt_positions)):
        # Query only previous points, not the current or future
        dists, idxs = tree.query(gt_positions[j], k=len(gt_positions), distance_upper_bound=threshold)
        
        # Filter valid indices (less than j and at least min_gap away)
        valid = [i for i in idxs if i < j - min_gap and i != len(gt_positions)]
        if valid:
            return valid[0], j  # Return first loop closure found

    return None, None  # No loop closure found


class BrisbaneEventDataset(BaseDataset):
    def __init__(self, base_path, sensor_size=(346, 260, 2)):
        self.base_path = base_path
        self.sensor_size = sensor_size
        self.video_beginning = {
            'sunset1': 1587452582.35,
            'sunset2': 1587540271.65,
            'daytime': 1587705130.80,
            'morning': 1588029265.73,
            'sunrise': 1588105232.91,
            'night': 1587975221.10,
            'sunset1_training': 1587452582.35,
            'sunset2_training': 1587540271.65,
            'daytime_training': 1587705130.80,
            'morning_training': 1588029265.73,
            'sunrise_training': 1588105232.91,
            'night_training': 1587975221.10
        }


    def load_events(self, sequence_name):
        import pyarrow as pa
        import pyarrow.parquet as pq
        # Open Parquet file via PyArrow
        events_path = os.path.join(self.base_path, 'paraquet_data', sequence_name, "events.parquet")
        parquet_file = pq.ParquetFile(events_path)

        # Collect batches
        t_list, x_list, y_list, p_list = [], [], [], []
        tqdm_stream = sys.stdout  # or sys.stderr if preferred
        for batch in tqdm(parquet_file.iter_batches(columns=['t', 'x', 'y', 'p'], batch_size=10_000_000), file=tqdm_stream, dynamic_ncols=True):
        # for batch in tqdm(,  desc="Loading paraquet"):
            table = pa.Table.from_batches([batch])
            df = table.to_pandas()
            t = df['t'].to_numpy()
            x = df['x'].to_numpy()
            y = df['y'].to_numpy()
            p = df['p'].fillna(0).astype(int).to_numpy()  # fill NaNs and convert safely
            
            t_list.append(t)
            x_list.append(x)
            y_list.append(y)
            p_list.append(p)
            

        # Concatenate all chunks
        t_all = np.concatenate(t_list)
        x_all = np.concatenate(x_list)
        y_all = np.concatenate(y_list)
        p_all = np.concatenate(p_list)

        # Create structured record array
        ev_dtype = np.dtype([('t', np.float64), ('x', int), ('y', int), ('p', int)])
        events = np.core.records.fromarrays([t_all, x_all, y_all, p_all], dtype=ev_dtype)
        return events


    def load_gps(self, sequence_name, args):
        gps_path = os.path.join(self.base_path, 'paraquet_data',  sequence_name, f"{sequence_name}_gps.txt")
        gps = np.loadtxt(gps_path)
        lat_lon = np.array(gps[:, :2])
        timestamps = gps[:, 2]
        return lat_lon, timestamps  # positions, timestamps


    def get_slice_indices(self, ev_ts: np.ndarray, gps_ts: np.ndarray, time_res: float, num_slices: int):
        """Get start and end indices of event timestamps for multiple slices within each time bin."""
        slice_duration = time_res / num_slices

        all_start_indices = []
        all_end_indices = []

        for t in gps_ts:
            for i in range(num_slices):
                slice_start = t + i * slice_duration
                slice_end = slice_start + slice_duration
                start_idx = np.searchsorted(ev_ts, slice_start, side='left')
                end_idx = np.searchsorted(ev_ts, slice_end, side='right')
                all_start_indices.append(start_idx)
                all_end_indices.append(end_idx)

        return np.array(all_start_indices), np.array(all_end_indices)
        

    def compute_headings(self, utm_coords, motion_threshold=1e-3):
        """
        Compute headings from UTM coordinates with improved handling of stationary periods.
        
        Args:
            utm_coords: Array of [x, y] coordinates
            motion_threshold: Minimum distance to consider as actual motion
        
        Returns:
            headings: Array of headings in degrees, same length as input
        """
        if len(utm_coords) < 2:
            return np.zeros(len(utm_coords))
        
        deltas = np.diff(utm_coords, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        
        # Compute raw headings only where there's significant motion
        raw_headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        
        # Create output array
        headings = np.zeros(len(utm_coords))
        
        # Strategy 1: Forward fill valid headings for stationary periods
        valid_motion = distances >= motion_threshold
        
        if np.any(valid_motion):
            # Set headings where there's valid motion
            headings[1:][valid_motion] = raw_headings[valid_motion]
            
            # Forward fill the first valid heading to the beginning
            first_valid_idx = np.where(valid_motion)[0][0] + 1  # +1 because we're indexing into headings
            if first_valid_idx < len(headings):
                headings[0] = headings[first_valid_idx]
            
            # Forward fill for stationary periods
            last_valid_heading = headings[0]
            for i in range(1, len(headings)):
                if valid_motion[i-1]:  # i-1 because valid_motion is one element shorter
                    last_valid_heading = headings[i]
                else:
                    headings[i] = last_valid_heading
        else:
            # No significant motion detected, set all headings to 0
            headings[:] = 0.0
        
        return headings


    def compute_angular_velocity(self, headings, times, window_size=100):
        """
        Compute smoothed angular velocity using heading difference over a sliding window.
        
        Args:
            headings: Array of heading angles in radians
            times: Array of timestamps
            window_size: Number of timesteps to use (e.g., 100 = 1s for dt=0.01)
        
        Returns:
            angular_velocities: Array of angular velocities in rad/s
        """
        dt = np.mean(np.diff(times))  # Should be ~0.01s
        unwrapped_headings = np.unwrap(headings)
        
        angular_velocities = np.zeros_like(headings)
        
        for i in range(window_size+1,len(headings)):
            i_start = max(0, i - window_size)
            delta_heading = unwrapped_headings[i] - unwrapped_headings[i_start]
            delta_time = times[i] - times[i_start]
            # delta_time = max(delta_time, 1e-6)  # Avoid division by zero
            angular_velocities[i] = delta_heading / delta_time
        angular_velocities=np.clip(angular_velocities, -np.pi/6, np.pi/6, out=angular_velocities)  # Clip to [-pi, pi]
        
        return angular_velocities


    def compute_speeds(self, utm_coords, times):
        """
        Compute instantaneous speeds from UTM coordinates and timestamps.
        
        Args:
            utm_coords: Array of shape (N, 2) with UTM coordinates
            times: Array of shape (N,) with timestamps
            
        Returns:
            speeds: Array of shape (N,) with speeds in m/s
        """
        # Compute distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(utm_coords, axis=0)**2, axis=1))
        
        # Compute time differences
        time_diffs = np.diff(times)
        
        # Avoid division by zero
        time_diffs = np.maximum(time_diffs, 1e-6)
        
        # Compute speeds
        speeds = distances / time_diffs
        
        # Pad with first speed value to maintain same length as input
        speeds = np.concatenate([[speeds[0]], speeds])
        
        return speeds, distances


    def compute_acceleration(self, speeds, angular_velocities, uniform_times, window_size=100):
        """
        Alternative acceleration computation using local polynomial fitting.
        This is more robust to noise than simple gradient.
        
        Args:
            speeds: Speed array
            angular_velocities: Angular velocity array
            uniform_times: Time array
            window_size: Size of local fitting window
        
        Returns:
            linear_acc, angular_acc: Acceleration arrays
        """
        n = len(speeds)
        linear_acc = np.zeros(n)
        angular_acc = np.zeros(n)
        
        half_window = window_size // 2
        
        for i in range(n):
            # Define window bounds
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            # Extract local data
            local_times = uniform_times[start:end] - uniform_times[i]  # Center around 0
            local_speeds = speeds[start:end]
            local_angular_vels = angular_velocities[start:end]
            
            if len(local_times) >= 3:
                # Fit local polynomial (degree 2 for acceleration)
                try:
                    # Linear acceleration
                    coeffs_linear = np.polyfit(local_times, local_speeds, min(2, len(local_times)-1))
                    if len(coeffs_linear) >= 2:
                        linear_acc[i] = coeffs_linear[-2]  # Coefficient of t^1 term
                    
                    # Angular acceleration  
                    coeffs_angular = np.polyfit(local_times, local_angular_vels, min(2, len(local_times)-1))
                    if len(coeffs_angular) >= 2:
                        angular_acc[i] = coeffs_angular[-2]  # Coefficient of t^1 term
                        
                except np.linalg.LinAlgError:
                    # Fallback to simple difference if fitting fails
                    if i > 0:
                        dt = uniform_times[i] - uniform_times[i-1]
                        linear_acc[i] = (speeds[i] - speeds[i-1]) / dt
                        angular_acc[i] = (angular_velocities[i] - angular_velocities[i-1]) / dt
        
        return linear_acc, angular_acc      

 
    def process_sequence(self, args, reforqry, reconstructor, gt_pos_only=False):
        # --- Determine sequence name ---
        if reforqry == 'ref':
            sequence_name = args.sequences[args.ref_seq_idx]
        elif reforqry == 'qry':
            sequence_name = args.sequences[args.qry_seq_idx]
        else:
            raise ValueError("reforqry must be 'ref' or 'qry'")

        # --- Load GPS data ---
        is_training_seq = "_training" in sequence_name
        latlon, gps_times = self.load_gps(sequence_name, args)
        if len(latlon) == 0 or len(gps_times) == 0:
            print("No GPS data found.")
            return None

        if not is_training_seq:
            gps_times += self.video_beginning[sequence_name]
            utm_coords = convert_latlon_to_utm(latlon)
            loop_id_start, loop_id_end = find_first_self_loop(utm_coords, threshold=10.0, min_gap=20)
            gps_times = gps_times[loop_id_start:loop_id_end + 1]
            utm_coords = utm_coords[loop_id_start:loop_id_end + 1]
            interp_x = interp1d(gps_times, utm_coords[:, 0], bounds_error=False, fill_value="extrapolate")
            interp_y = interp1d(gps_times, utm_coords[:, 1], bounds_error=False, fill_value="extrapolate")

            uniform_times = np.arange(gps_times[0], gps_times[-1], args.min_time_res)
            utm_interp = np.stack([interp_x(uniform_times), interp_y(uniform_times)], axis=1)
        else:
            # Convert lat/lon to UTM
            utm_coords = convert_latlon_to_utm(latlon)
            # CROP START/END
            crop_margin_us = 25  # 5 seconds in microseconds
            start_time = gps_times[0] + crop_margin_us
            end_time = gps_times[-1] - crop_margin_us
            keep_mask = (gps_times >= start_time) & (gps_times <= end_time)
            gps_times = gps_times[keep_mask]

            gps_times += self.video_beginning[sequence_name]
            utm_coords = utm_coords[keep_mask]

            # Segment based on large time gaps
            gps_time_diffs = np.diff(gps_times)
            gap_indices = np.where(gps_time_diffs > 5e6)[0]  # adjust threshold if needed
            segment_boundaries = np.concatenate([[0], gap_indices + 1, [len(gps_times)]])

            # Interpolate each segment
            segments = []
            for i in range(len(segment_boundaries) - 1):
                start, end = segment_boundaries[i], segment_boundaries[i+1]
                if end - start < 2:
                    continue
                times_seg = gps_times[start:end]
                utm_seg = utm_coords[start:end]
                interp_x = interp1d(times_seg, utm_seg[:, 0], bounds_error=False, fill_value=np.nan)
                interp_y = interp1d(times_seg, utm_seg[:, 1], bounds_error=False, fill_value=np.nan)
                segments.append((times_seg[0], times_seg[-1], interp_x, interp_y))

            # Create uniformly sampled times and interpolate UTM
            uniform_times = np.arange(gps_times[0], gps_times[-1], args.min_time_res)
            utm_interp = np.full((len(uniform_times), 2), np.nan)
            for t_start, t_end, interp_x, interp_y in segments:
                seg_mask = (uniform_times >= t_start) & (uniform_times <= t_end)
                utm_interp[seg_mask, 0] = interp_x(uniform_times[seg_mask])
                utm_interp[seg_mask, 1] = interp_y(uniform_times[seg_mask])

        # --- Compute dynamics ---
        window_size = int(1 / args.min_time_res)
        headings = self.compute_headings(utm_interp)
        ang_vel = self.compute_angular_velocity(headings, uniform_times, window_size)
        speed, displacements = self.compute_speeds(utm_interp, uniform_times)
        lin_acc, ang_acc = self.compute_acceleration(speed, ang_vel, uniform_times, window_size)

        # --- Bin sequence ---
        if args.count_bin == 1:
            events = self.load_events(sequence_name)
            t_events = events['t']
            n_events = len(t_events)
            events_per_bin = args.events_per_bin  # You must add this argument externally

            start_idxs = np.arange(0, n_events, events_per_bin)
            end_idxs = np.minimum(start_idxs + events_per_bin, n_events)

            bin_starts = t_events[start_idxs]
            bin_ends = t_events[end_idxs - 1]
            bin_durs = bin_ends - bin_starts
            bin_mids = (bin_starts + bin_ends) / 2

            # Use bin start times to compute GT positions
            gt_positions = np.stack([interp_x(bin_starts), interp_y(bin_starts)], axis=1)

            print(f"Event-count binning with {events_per_bin} events per bin. Total bins: {len(gt_positions)}")
        else:
            # Fixed binning
            bin_size = args.time_res
            bin_starts = np.arange(uniform_times[0], uniform_times[-1] - bin_size, bin_size)
            bin_ends = bin_starts + bin_size
            bin_durs = np.full_like(bin_starts, bin_size)
            bin_mid_idxs = np.searchsorted(uniform_times, (bin_starts + bin_ends) / 2)
            bin_speeds = speed[bin_mid_idxs]
            bin_ang_vels = ang_vel[bin_mid_idxs]

        
        if args.count_bin != 1:
            # # --- Get GT positions ---
            gt_positions = np.stack([interp_x(bin_starts), interp_y(bin_starts)], axis=1)

            print(f"Bin durations: {np.unique(bin_durs)} (s), total bins: {len(gt_positions)}")
            assert np.allclose(bin_starts[1:], bin_ends[:-1], atol=1e-8), "Bins overlap or have gaps"

            # --- Plot dynamics ---
            plot_vels_accs_bins(bin_starts, bin_durs, bin_speeds, bin_ang_vels,
                                uniform_times, lin_acc, ang_acc, gt_positions[:, 0], gt_positions[:, 1], sequence_name)


        # --- Load and slice events ---
        events = self.load_events(sequence_name)
        t_events = events['t']
        start_idxs = np.searchsorted(t_events, bin_starts, side='left')
        end_idxs = np.searchsorted(t_events, bin_ends, side='right')

        # Create mask for valid (non-empty) bins where start != end
        valid_mask = start_idxs != end_idxs

        # Apply mask to keep only valid pairs
        start_idxs = start_idxs[valid_mask]
        end_idxs = end_idxs[valid_mask]


        # --- Reconstruct event frames ---
        array_3d, _ = reconstructor.reconstruct(
            eventsData=events,
            sensor_size=self.sensor_size,
            start_indices=start_idxs,
            end_indices=end_idxs,
            hp_loc=os.path.join(self.base_path, 'paraquet_data', sequence_name, 'hot_pixels.txt')
        )

        return np.array(array_3d), np.array(gt_positions)


    def save_training_split(self, args):
        """
        Save the training split for the Brisbane dataset.
        
        """
        import pyarrow.parquet as pq
        import matplotlib.pyplot as plt
        import pandas as pd

        for seq in args.sequences:
            print(f"Processing sequence {seq}")
            latlon, gps_times_loaded = self.load_gps(seq, args)
            if len(latlon) == 0 or len(gps_times_loaded) == 0:
                print("No GPS data found.")
                return None

            gps_times = gps_times_loaded+self.video_beginning[seq]
            utm_coords = convert_latlon_to_utm(latlon)

            loop_id_start, loop_id_end = find_first_self_loop(utm_coords, threshold=10.0, min_gap=20)

            # Save GPS data before loop start and after loop end
            if loop_id_start is not None and loop_id_end is not None:
                # Combine GPS data before loop and after loop
                latlon_before = latlon[:loop_id_start]
                gps_times_before = gps_times_loaded[:loop_id_start]
                
                latlon_after = latlon[loop_id_end:]
                gps_times_after = gps_times_loaded[loop_id_end:]
                
                # Concatenate before and after data
                latlon_training = np.concatenate([latlon_before, latlon_after])
                gps_times_training = np.concatenate([gps_times_before, gps_times_after])
                
                # Combine into single array [lat, lon, time]
                gps_training_data = np.column_stack([latlon_training, gps_times_training])
                
                # Save training GPS data
                training_gps_path = os.path.join(self.base_path, 'paraquet_data', f"{seq}_training", f"{seq}_training_gps.txt")
                os.makedirs(os.path.dirname(training_gps_path), exist_ok=True)
                np.savetxt(training_gps_path, gps_training_data, fmt='%.8f')
                print(f"Saved training GPS data to {training_gps_path}")
            else:
                print(f"No loop closure found for {seq}, using full GPS data")

            # --- Load and slice events ---
            loop_start_time = gps_times[loop_id_start]
            loop_end_time = gps_times[loop_id_end]
            events = self.load_events(seq)
            t_events = events['t']

            print(f"Loop start time: {loop_start_time}, Loop end time: {loop_end_time}")
            print(f"maximum event time: {np.max(t_events)} minimum event time: {np.min(t_events)}")

            # Find indices for slicing
            before_loop_idx = np.searchsorted(t_events, loop_start_time, side='left')
            after_loop_idx = np.searchsorted(t_events, loop_end_time, side='right')

            print(f"Before loop index: {before_loop_idx}, After loop index: {after_loop_idx}")

            # Combine events before the loop and after the loop
            events_before = events[:before_loop_idx]
            events_after = events[after_loop_idx:]
            events_slice = np.concatenate([events_before, events_after])

            print(f" lenght of events from {len(events)} -> {len(events_slice)} after slicing")

            # Create output path for training events
            training_seq_name = seq + '_training'
            events_path = os.path.join(self.base_path, 'paraquet_data', training_seq_name, "events.parquet")
            os.makedirs(os.path.dirname(events_path), exist_ok=True)

            # Convert to DataFrame and save
            df_events = pd.DataFrame({
                't': events_slice['t'],
                'x': events_slice['x'],
                'y': events_slice['y'],
                'p': events_slice['p']
            })
            table = pa.Table.from_pandas(df_events)
            pq.write_table(table, events_path)
            print(f"Saved training event slice to {events_path}")

            assert False





class Brisbane_RGB_Dataset(BaseDataset):
    def __init__(self, base_path):
        super().__init__(base_path)
        self.video_beginning = {
            'sunset1': 1587452582.35,
            'sunset2': 1587540271.65,
            'daytime': 1587705130.80,
            'morning': 1588029265.73,
            'sunrise': 1588105232.91,
            'night': 1587975221.10
        }
        self.sequences = ['sunset1', 'sunset2', 'daytime', 'night', 'morning', 'sunrise']              # list: ["sunset1", "sunset2", ...]
        # self.tonic_vpr = TonicVPR(save_to=os.path.join(base_path, "tonic2"))  # init once



    def read_gps_file(self, nmea_file_path, vid_beg_shift = False):
        """
        Read GPS data from NMEA file and return lat/lon/timestamp array.
        """
        from datetime import datetime, timezone
        from geopy.distance import geodesic
        nmea_file = open(nmea_file_path, encoding="utf-8")
        latitudes, longitudes, timestamps = [], [], []
        previous_lat, previous_lon = 0, 0
        first_timestamp = None
        if vid_beg_shift == True:
            for name in self.video_beginning:
                if name in nmea_file_path:
                    first_timestamp = self.video_beginning[name]
                    first_timestamp = datetime.fromtimestamp(first_timestamp, tz=timezone.utc).time()
                    break
        
        for line in nmea_file.readlines():
            try:
                msg = pynmea2.parse(line)
                if first_timestamp is None:
                    first_timestamp = msg.timestamp
                if msg.sentence_type not in ["GSV", "VTG", "GSA"]:
                    if (msg.latitude != 0 and msg.longitude != 0 and 
                        msg.latitude != previous_lat and msg.longitude != previous_lon):
                        dist_to_prev = geodesic((msg.latitude, msg.longitude),
                                            (previous_lat, previous_lon)).meters
                        if dist_to_prev > 1.0:
                            timestamp_diff = (
                                (msg.timestamp.hour - first_timestamp.hour) * 3600
                                + (msg.timestamp.minute - first_timestamp.minute) * 60
                                + (msg.timestamp.second - first_timestamp.second)
                                + (msg.timestamp.microsecond - first_timestamp.microsecond) / 1e6
                            )
                            latitudes.append(msg.latitude)
                            longitudes.append(msg.longitude)
                            timestamps.append(timestamp_diff)
                            previous_lat, previous_lon = msg.latitude, msg.longitude

            except pynmea2.ParseError as e:
                continue
        
        nmea_file.close()
        return np.array(np.vstack((latitudes, longitudes, timestamps))).T


    def load_gps_data(self,base_path, seq_name, vid_beg_shift=False):
        nmea_path = os.path.join(base_path, 'rgb_video', f"{seq_name}.nmea")
        gps_data = self.read_gps_file(nmea_path, vid_beg_shift)
        if len(gps_data) == 0:
            print(f"No GPS data found for sequence {seq_name}.")
            return None, None
        lat_lon, gps_times_sec = gps_data[:, :2], gps_data[:, 2]  # lat, lon, timestamp
        gps_times_us = gps_times_sec * 1e6  # convert to microseconds

        return lat_lon, gps_times_sec
    

    def frame_generator(self, filepath, crop_height=40, every_nth_frame=1):
        import cv2
        cap = cv2.VideoCapture(filepath)
        frame_idx = 0

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {filepath}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_nth_frame == 0:
                height = frame.shape[0]
                cropped = frame[:height - crop_height]
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield cropped, timestamp

            frame_idx += 1

        cap.release()


    def process_sequence(self, args, reforqry, reconstructor):
        # --- Interpolate GPS positions ---
        seq_name = args.sequences[args.ref_seq_idx] if reforqry == 'ref' else args.sequences[args.qry_seq_idx]
        lat_lon, gps_times_sec = self.load_gps_data(self.base_path, seq_name, vid_beg_shift=False)
        print(f"Loaded {len(lat_lon)} positions from GPS data for sequence {seq_name}")
        print(f" GPS start/end (s): {gps_times_sec[0]} -> {gps_times_sec[-1] }" )
        utm_coords = convert_latlon_to_utm(lat_lon)

        #load video
        
        image_frames = []
        image_timestamps = []
        filepath = os.path.join(self.base_path, 'rgb_video', f"{seq_name}.mp4")
        for frame, ts in self.frame_generator(filepath, crop_height=100, every_nth_frame=10):
            image_frames.append(frame)
            image_timestamps.append(ts)
            # print(f"Frame timestamp: {ts:.2f} seconds")
        print(f"Loaded {len(image_frames)} frames from video {seq_name}")
        print(f"image before start/end (s): {image_timestamps[0]} -> {image_timestamps[-1]}" )


        # Interpolate GPS positions
        start_time = max(gps_times_sec[0], image_timestamps[0])  # safe bound
        end_time = min(gps_times_sec[-1], image_timestamps[-1])  # safe bound
        interp_times = np.arange(start_time, end_time, args.time_res)
        interp_x = interp1d(gps_times_sec, utm_coords[:, 0], kind='linear', fill_value="extrapolate")
        interp_y = interp1d(gps_times_sec, utm_coords[:, 1], kind='linear', fill_value="extrapolate")

        interp_positions = np.stack([
            interp_x(interp_times),
            interp_y(interp_times)
        ], axis=1)
        
        # Find closest video frame for each interpolated GPS time
        image_timestamps_np = np.array(image_timestamps)
        closest_frame_idxs = np.argmin(np.abs(image_timestamps_np[:, None] - interp_times[None, :]), axis=0)
        matched_frames = [image_frames[i] for i in closest_frame_idxs]


        return  matched_frames, interp_positions



# elif args.adaptive_bin == 1:
#     bin_starts, bin_ends, bin_durs, bin_speeds, bin_ang_vels = [], [], [], [], []
#     i = 0
#     while i < len(uniform_times) - 1:
#         cur_vals = (ang_vel[i], speed[i], ang_acc[i], lin_acc[i])
#         bin_size = max(1, adaptive_bin_size(*cur_vals, ang_acc_max=args.max_odom[0],
#                                             ang_vel_max=args.max_odom[1], 
#                                             lin_speed_max=args.max_odom[2],
#                                             lin_acc_max=args.max_odom[3],
#                                             max_bins=args.max_bins, weights=args.odom_weights, 
#                                             use_exponential=args.use_exponential))
#         i_end = int(min(i + bin_size, len(uniform_times) - 1))
#         bin_starts.append(uniform_times[i])
#         bin_ends.append(uniform_times[i_end])
#         bin_durs.append(uniform_times[i_end] - uniform_times[i])
#         bin_speeds.append(speed[i])
#         bin_ang_vels.append(ang_vel[i])
#         i = i_end
#     bin_starts = np.array(bin_starts)
#     bin_ends = np.array(bin_ends)
#     bin_durs = np.array(bin_durs)