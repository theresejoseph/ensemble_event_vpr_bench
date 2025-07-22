# datasets/nsvap.py
from datasets.base_dataset import BaseDataset
import h5py
from utils.odometry_utils import compute_acceleration, compute_speeds, compute_headings, compute_angular_velocity, adaptive_bin_size, plot_vels_accs_bins
import numpy as np
import os 
import cv2
from scipy.interpolate import interp1d
import numpy as np
import os


class NSVAPDataset(BaseDataset):
    def __init__(self, base_path, sensor_size=(640, 480, 2), data_fraction=1):
        self.base_path = base_path
        self.sensor_size = sensor_size
        self.data_fraction = data_fraction
        self.processed_dir = os.path.join(base_path, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_events(self, sequence_name):
        events_path = os.path.join(self.base_path, "h5_data", f"{sequence_name}_dvxplorer_left.h5")
        with h5py.File(events_path, "r") as f:
            n = int(self.data_fraction * f["events/timestamps"].shape[0])
            t = f["events/timestamps"][:n] * 1e-9  # ns → sec
            x = f["events/x_coordinates"][:n]
            y = f["events/y_coordinates"][:n]
            p = f["events/polarities"][:n]
        return np.core.records.fromarrays([t, x, y, p], names='t,x,y,p')


    def load_gps(self, sequence_name):
        events_path = os.path.join(self.base_path, "h5_data", f"{sequence_name}_dvxplorer_left.h5")
        gps_path = os.path.join(self.base_path, "h5_data", f"{sequence_name}_applanix.h5")

        with h5py.File(events_path, "r") as f:
            event_timestamps = f["events/timestamps"]
            n = int(self.data_fraction * event_timestamps.shape[0])
            start_time = event_timestamps[0] * 1e-9
            end_time = event_timestamps[n - 1] * 1e-9

        with h5py.File(gps_path, "r") as f:
            gps_times = f["pose_base_link/timestamps"][:] * 1e-9
            gps_pos = f["pose_base_link/positions"][:, :2]

        # Crop GPS to match event time range
        valid_mask = (gps_times >= start_time) & (gps_times <= end_time)
        return gps_pos[valid_mask], gps_times[valid_mask]


    def process_sequence(self, args, reforqry, reconstructor):
        sequence_name = args.sequences[args.ref_seq_idx if reforqry == 'ref' else args.qry_seq_idx]
        gps_pos, gps_times = self.load_gps(sequence_name)
        events_path = os.path.join(self.base_path, "h5_data", f"{sequence_name}_dvxplorer_left.h5")

        interp_x = interp1d(gps_times, gps_pos[:, 0], bounds_error=False, fill_value="extrapolate")
        interp_y = interp1d(gps_times, gps_pos[:, 1], bounds_error=False, fill_value="extrapolate")
        uniform_times = np.arange(gps_times[0], gps_times[-1], args.min_time_res)
        utm_interp = np.stack([interp_x(uniform_times), interp_y(uniform_times)], axis=1)

        window_size = int(1 / args.min_time_res)
        headings = compute_headings(utm_interp)
        ang_vel = compute_angular_velocity(headings, uniform_times, window_size)
        speed = compute_speeds(utm_interp, uniform_times)
        lin_acc, ang_acc = compute_acceleration(speed, ang_vel, uniform_times, window_size)

        # --- Bin generation ---
        if args.count_bin == 1:
            with h5py.File(events_path, "r") as f:
                t_dset = f["events/timestamps"]
                total_events = t_dset.shape[0]
                usable_events = int(total_events * self.data_fraction)

                duration_s = (t_dset[usable_events - 1] - t_dset[0]) * 1e-9
                print(f"Traverse duration: {duration_s / 60:.1f} minutes", flush=True)

                events_per_bin = int(args.events_per_bin)
                start_idxs = np.arange(0, usable_events, events_per_bin)
                end_idxs = np.minimum(start_idxs + events_per_bin, usable_events)

                bin_starts = t_dset[start_idxs] * 1e-9
                bin_ends = t_dset[end_idxs - 1] * 1e-9

            bin_durs = bin_ends - bin_starts
            bin_mids = (bin_starts + bin_ends) / 2
            gt_positions = np.stack([interp_x(bin_mids), interp_y(bin_mids)], axis=1)

        elif args.adaptive_bin == 1:
            bin_starts, bin_ends, bin_durs, bin_speeds, bin_ang_vels = [], [], [], [], []
            i = 0
            while i < len(uniform_times) - 1:
                cur_vals = (ang_vel[i], speed[i], ang_acc[i], lin_acc[i])
                bin_size = max(1, adaptive_bin_size(*cur_vals, ang_acc_max=args.max_odom[0],
                                                    ang_vel_max=args.max_odom[1], 
                                                    lin_speed_max=args.max_odom[2],
                                                    lin_acc_max=args.max_odom[3],
                                                    max_bins=args.max_bins, weights=args.odom_weights, 
                                                    use_exponential=args.use_exponential))
                i_end = int(min(i + bin_size, len(uniform_times) - 1))
                bin_starts.append(uniform_times[i])
                bin_ends.append(uniform_times[i_end])
                bin_durs.append(uniform_times[i_end] - uniform_times[i])
                bin_speeds.append(speed[i])
                bin_ang_vels.append(ang_vel[i])
                i = i_end

            bin_starts = np.array(bin_starts)
            bin_ends = np.array(bin_ends)
            bin_mids = (bin_starts + bin_ends) / 2
            gt_positions = np.stack([interp_x(bin_starts), interp_y(bin_starts)], axis=1)

            assert np.allclose(bin_starts[1:], bin_ends[:-1], atol=1e-8), "Bins overlap or have gaps"

            plot_vels_accs_bins(bin_starts, bin_durs, bin_speeds, bin_ang_vels,
                                uniform_times, lin_acc, ang_acc,
                                gt_positions[:, 0], gt_positions[:, 1], sequence_name)

        else:
            bin_size = args.time_res
            bin_starts = np.arange(uniform_times[0], uniform_times[-1] - bin_size, bin_size)
            bin_ends = bin_starts + bin_size
            bin_mids = (bin_starts + bin_ends) / 2
            gt_positions = np.stack([interp_x(bin_starts), interp_y(bin_starts)], axis=1)
        print(f"Bin durations: {np.unique(bin_ends - bin_starts)} (s), total bins: {len(bin_starts)}", flush=True)
        
        
        # --- Trim to only bins after the first movement ---
        movement_mask = speed > 0
        if not np.any(movement_mask):
            raise ValueError(f"No movement detected in sequence {sequence_name}; cannot reconstruct.")
        first_moving_time = uniform_times[np.argmax(movement_mask)]
        valid_bin_mask = bin_starts >= first_moving_time
        bin_starts = bin_starts[valid_bin_mask]
        bin_ends = bin_ends[valid_bin_mask]
        bin_mids = bin_mids[valid_bin_mask]
        gt_positions = gt_positions[valid_bin_mask]

        import matplotlib.pyplot as plt
        plt.plot(gps_pos[:, 0], gps_pos[:, 1], 'k-', label='GPS Path')
        plt.savefig(f"./plots/{sequence_name}_gps_path.png")
        plt.close()

                
        # --- Reconstruct ---
        events = self.load_events(sequence_name)
        t_events = events['t']

        if args.count_bin != 1:
            start_idxs = np.searchsorted(t_events, bin_starts, side='left')
            end_idxs = np.searchsorted(t_events, bin_ends, side='right')
        else:
            start_idxs = start_idxs[valid_bin_mask]
            end_idxs = end_idxs[valid_bin_mask]
        print(f"Reconstructing {len(start_idxs)} bins for sequence {sequence_name}", flush=True)

        frames, _ = reconstructor.reconstruct(
            eventsData=events,
            sensor_size=self.sensor_size,
            start_indices=start_idxs,
            end_indices=end_idxs)
        


        return np.array(frames), gt_positions
    
class NSAVP_RGB_Dataset(BaseDataset):
    def __init__(self, base_path):
        super().__init__(base_path)

    def process_sequence(self, args, reforqry, reconstructor):
        """Process RGB sequence and extract images with interpolated GT positions (at ~10 Hz)."""
        if reforqry == 'ref':
            sequence_name = args.sequences[args.ref_seq_idx]
        elif reforqry == 'qry':
            sequence_name = args.sequences[args.qry_seq_idx]

        print(f"Processing RGB sequence {sequence_name}", flush=True)
        rgb_path = f"{self.base_path}/h5_data/{sequence_name}_rgb_left.h5"
        gt_path = f"{self.base_path}/h5_data/{sequence_name}_applanix.h5"

        # Load ground truth (pose)
        with h5py.File(gt_path, "r") as f:
            gt_timestamps = f["pose_base_link/timestamps"][:].astype(np.float64) * 1e-9  # ns -> sec
            gt_positions = f["pose_base_link/positions"][:]  # shape: (N, 3)

        # Load images and timestamps
        with h5py.File(rgb_path, "r") as f:
            all_timestamps = f["image_raw/timestamps"][:].astype(np.float64) * 1e-9  # ns -> sec

            selected_indices = [0]
            last_ts = all_timestamps[0]
            for i in range(1, len(all_timestamps)):
                if all_timestamps[i] - last_ts >= args.time_res:  # ≥ 100 ms
                    selected_indices.append(i)
                    last_ts = all_timestamps[i]

            image_timestamps = all_timestamps[selected_indices]

            images = []
            for idx in selected_indices:
                img = f["image_raw/images"][idx]
                if "bayer" in f["image_raw"].attrs.get("encoding", "").lower():
                    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                images.append(img)
            images = np.array(images)

        # Interpolate GT for selected timestamps
        interpolated_positions = []
        for i in range(3):  # x, y, z
            interp_func = interp1d(gt_timestamps, gt_positions[:, i], bounds_error=False, fill_value="extrapolate")
            interpolated = interp_func(image_timestamps)
            interpolated_positions.append(interpolated)
        interpolated_positions = np.stack(interpolated_positions, axis=1)

        return images, interpolated_positions


    # def process_sequence(self, args, reforqry, reconstructor):
    #     sequence_name = args.sequences[args.ref_seq_idx if reforqry == 'ref' else args.qry_seq_idx]
    #     gps_pos, gps_times = self.load_gps(sequence_name)
    #     events_path = os.path.join(self.base_path, "h5_data", f"{sequence_name}_dvxplorer_left.h5")

    #     interp_x = interp1d(gps_times, gps_pos[:, 0], bounds_error=False, fill_value="extrapolate")
    #     interp_y = interp1d(gps_times, gps_pos[:, 1], bounds_error=False, fill_value="extrapolate")
    #     uniform_times = np.arange(gps_times[0], gps_times[-1], args.min_time_res)

    #     bin_size = args.time_res
    #     bin_starts = np.arange(uniform_times[0], uniform_times[-1] - bin_size, bin_size)
    #     bin_ends = bin_starts + bin_size
    #     gt_positions = np.stack([interp_x(bin_starts), interp_y(bin_starts)], axis=1)
    #     # --- Reconstruct ---
    #     events = self.load_events(sequence_name)
    #     t_events = events['t']
    #     start_idxs = np.searchsorted(t_events, bin_starts, side='left')
    #     end_idxs = np.searchsorted(t_events, bin_ends, side='right')
    #     print(f"Reconstructing {len(start_idxs)} bins for sequence {sequence_name}", flush=True)

    #     frames, _ = reconstructor.reconstruct(
    #         eventsData=events,
    #         sensor_size=self.sensor_size,
    #         start_indices=start_idxs,
    #         end_indices=end_idxs)


    #     return np.array(frames), gt_positions