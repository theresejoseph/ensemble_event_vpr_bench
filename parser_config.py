import argparse

default_config = {
    'NSAVP': {
        'sequences': ["R0_FA0", "R0_FS0", "R0_FN0", "R0_RA0", "R0_RS0", "R0_RN0"],
        'ref_seq_idx': 10,
        'qry_seq_idx':11,
        'reconstruct_method_name': 'eventCount',
        'metric_data_spacing':10,
        'num_samples_per_location': 1, #range {1,2,5,10}
        'min_dist_toler': 10,
        'time_res': 0.1,
        'min_time_res': 0.01,
        'adaptive_bin': 0,
        'dataset_path': "/mnt/hpccs01/home/n10234764/data/NSAVP",
        'start_idx': 0,
        'end_idx': None,
        'dist_thresh': 10,
        'patch': 1,
        'patch_rows': 2,
        'patch_cols': 3,
        'normalize_ref': 1,
        'normalize_qry': 1,
        'save_results': 0,
        'results_loc': '/mnt/hpccs01/home/n10234764/event_vpr/results/nsavp',
        'rerun_place_rec': 0,
        'evaluate_seq_mats': 0,
        'save_matches_video': 0,
        'patch_select': 'most_confident',
        'training_data_path':"/mnt/hpccs01/home/n10234764/event_vpr/results/nsavp/patch_feature_data/patch_EPR_data_R0_FA0vsR0_FN0_3400.csv",
        'exp_tag':'',
        'odom_weights': [0.4, 0.15, 0.4,0.05] ,
        'max_odom': [3.6, 20, 3.6 ,10],
        'adaptive_bin_tag': 'manual_adaptive',
        'max_bins': 20,
        'count_bin': 0,  
        'events_per_bin': 100_000,  # Number of events per bin for fixed count binning
        'save_frames_video': 1,
        'use_exponential': False,  # Use exponential binning if True, linear otherwise
    },
    
    'Brisbane': {
        'sequences':["night","morning", "sunrise", "sunset1", "sunset2", "daytime"],
        'ref_seq_idx': 1,
        'qry_seq_idx': 4,
        'reconstruct_method_name': 'eventCount',
        'metric_data_spacing':10,
        'num_samples_per_location': 1, #range {1,2,5,10}
        'min_dist_toler': 10,
        'time_res': 0.1,
        'min_time_res': 0.01,
        'max_bins': 50, 
        'adaptive_bin': 0,
        'dataset_path': "/mnt/hpccs01/home/n10234764/data/BrisbaneEvent",
        'start_idx': 0,
        'end_idx': None,
        'dist_thresh': 10,
        'patch': 1,
        'patch_rows': 3,
        'patch_cols': 2,
        'normalize_ref': 1,
        'normalize_qry': 1,
        'save_results': 0,
        'results_loc': '/mnt/hpccs01/home/n10234764/event_vpr/results/brisbane',
        'rerun_place_rec': 0,
        'evaluate_seq_mats': 0,
        'save_matches_video': 0,
        'patch_select': 'most_confident',
        'training_data_path':"/mnt/hpccs01/home/n10234764/event_vpr/results/nsavp/patch_feature_data/patch_EPR_data_R0_FA0vsR0_FN0_3400.csv", 
        'exp_tag':'',
        'odom_weights': [0.4, 0.15, 0.4,0.05] ,
        'max_odom': [3.6, 20, 3.6 ,10],
        'save_frames_video': 1,
        'adaptive_bin_tag': 'manual_adaptive',
        'count_bin': 0,  # Enable binning with fixed count
        'events_per_bin': 100_000,  # Number of events per bin for fixed count binning
        'use_exponential': False,  # Use exponential binning if True, linear otherwise
    },
    
    'NYC_Bench': {
        'sequences': ["event_25m_0sobel_1fps_day", "event_25m_0sobel_1fps_night", "event_5m_0sobel_1fps", "event_10m_0sobel_1fps", "event_25m_0sobel_1fps"],
        'ref_seq_idx': 1,   # Index of the sequence used for reference images.
        'qry_seq_idx': 1,   # Index of the sequence used for query images.
        'query_incr': 1,
        'ref_incr': 1,
        'min_dist_toler': 25,
        'time_res': 0.1,
        'dataset_path': "/mnt/hpccs01/home/n10234764/data/NYC-Event-VPR/NYC-Event-VPR_VPR-Bench",  # Replace with the real base path.
        'start_idx': 0,
        'end_idx': 500,
        'dist_thresh': 25,
        'patch': 0,
        'patch_rows': 3,
        'patch_cols': 2,
        'normalize_ref': 0,
        'normalize_qry': 0,
        'random_downsample_bool': 0,
        'downSampWindow': 20,
        'max_active_pixels': 150,
        'shift': 1,
        'max_row_shift': 0.02,
        'max_col_shift': 0.1,
        'latlong': 0,
        'save_results': 1,
        'results_loc': './results/NYC',
        'rerun_place_rec': 1,
        'evaluate_seq_mats': 0,
        'save_matches_video': 1,
        'patch_select': 'most_confident'
    },
    
    'NYC_eval': {
        'sequences': ["train", "test", "val"],
        'ref_seq_idx': 1,   # Index of the sequence used for reference images.
        'qry_seq_idx': 1,   # Index of the sequence used for query images.
        'query_incr': 1,
        'ref_incr': 1,
        'min_dist_toler': 25,
        'time_res': 0.1,
        'dataset_path': "/mnt/hpccs01/home/n10234764/data/NYC-Event-VPR/NYC-Event-VPR_VG",  # Replace with the real base path.
        'start_idx': 0,
        'end_idx': 0,
        'dist_thresh': 25,
        'patch': 1,
        'patch_rows': 3,
        'patch_cols': 2,
        'normalize_ref': 0,
        'normalize_qry': 0,
        'random_downsample_bool': 0,
        'downSampWindow': 20,
        'max_active_pixels': 150,
        'shift': 1,
        'max_row_shift': 0.02,
        'max_col_shift': 0.1,
        'latlong': 0,
        'save_results': 1,
        'results_loc': './results/NYC',
        'rerun_place_rec': 0,
        'evaluate_seq_mats': 1,
        'save_matches_video': 1,
        'patch_select': 'most_confident'
    },
    
    'NYC': {
        'sequences': ["sensor_data_2022-12-06_18-27-21",  "sensor_data_2022-12-07_16-52-55",  "sensor_data_2022-12-09_18-56-13",  "sensor_data_2023-02-14_15-06-30",
                        "sensor_data_2022-12-06_19-27-59",  "sensor_data_2022-12-07_17-58-34",  "sensor_data_2022-12-09_19-40-27",  "sensor_data_2023-02-14_18-20-40",
                        "sensor_data_2022-12-06_20-45-53",  "sensor_data_2022-12-09_13-59-10",  "sensor_data_2022-12-09_19-42-07",  "sensor_data_2023-04-20_15-53-26",
                        "sensor_data_2022-12-07_15-46-32",  "sensor_data_2022-12-09_14-41-29",  "sensor_data_2022-12-20_16-54-11",  "sensor_data_2023-04-20_17-10-01"],
        'ref_seq_idx': 0,   # Index of the sequence used for reference images.
        'qry_seq_idx': 1,   # Index of the sequence used for query images.
        'query_incr': 1,
        'ref_incr': 1,
        'min_dist_toler': 25,
        'time_res': 0.1,
        'dataset_path': "/mnt/hpccs01/home/n10234764/data/NYC-Event-VPR/NYC-Event-VPR_raw_data",  # Replace with the real base path.
        'start_idx': 0,
        'end_idx': 1000,
        'dist_thresh': 25,
        'patch': 1,
        'patch_rows': 3,
        'patch_cols': 2,
        'normalize_ref': 0,
        'normalize_qry': 0,
        'random_downsample_bool': 0,
        'downSampWindow': 20,
        'max_active_pixels': 150,
        'shift': 1,
        'max_row_shift': 0.02,
        'max_col_shift': 0.1,
        'latlong': 1,
        'save_results': 1,
        'results_loc': './results/NYC',
        'rerun_place_rec': 0,
        'evaluate_seq_mats': 1,
        'save_matches_video': 1,
        'patch_select': 'most_confident',
        'exp_tag':'',
    }
    # add other datasets as needed...
}

def get_parser():
    parser = argparse.ArgumentParser(description="Event-based Place Recognition")
    parser.add_argument("--dataset_type", type=str, default=None, help="Dataset type (e.g., NSAVP, Brisbane)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--sequences", type=list, default=None, help="Sequence names")
    parser.add_argument("--metric_data_spacing", type=float, default=None, help="Spacing between frames in the dataset")
    parser.add_argument('--ref_seq_idx', type=int, default=None, help='Picking reference sequence for evaluation')
    parser.add_argument('--qry_seq_idx', type=int, default=None, help='Picking query sequence for evaluation')
    parser.add_argument("--query_incr", type=int, default=None, help="Query increment")
    parser.add_argument("--ref_incr", type=int, default=None, help="Reference increment")
    parser.add_argument("--min_dist_toler", type=float, default=None, help="Minimum distance tolerance")
    parser.add_argument("--time_res", type=float, default=None, help="Time resolution")
    parser.add_argument("--data_type", type=str, default=None, help="Data type tag")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for processing")
    parser.add_argument("--end_idx", type=lambda x: None if x.lower() == 'none' else int(x), default=None, help="End index for processing")
    parser.add_argument("--dist_thresh", type=int, default=None, help="Distance threshold")
    parser.add_argument("--patch", type=int, default=None, help="Use patch-based matching")
    parser.add_argument('--patch_rows', type=int, default=None, help='Number of rows for patching')
    parser.add_argument('--patch_cols', type=int, default=None, help='Number of columns for patching')
    parser.add_argument('--patch_select', type=str, default=None, help='Selection method for patch')
    parser.add_argument('--patch_conf', type=float, default=None, help='Number of columns for patching')

    parser.add_argument('--normalize_ref', type=int, default=None, help='Normalize reference frames')
    parser.add_argument('--normalize_qry', type=int, default=None, help='Normalize query frames')
    parser.add_argument('--num_samples_per_location', type=int, default=None, help='Enable random downsampling')
    parser.add_argument('--reconstruct_method_name', type=str, default=None, help='GT positions format')

    parser.add_argument('--results_loc', type=str, default=None, help='Location of results')
    parser.add_argument('--save_results', type=int, default=None, help='Boolean for determining whether results should be saved')
    parser.add_argument('--save_frames_video', type=int, default=None, help='Boolean for determining whether results should be saved')
    parser.add_argument('--rerun_place_rec', type=int, default=None, help='Enable rerun place rec')
    parser.add_argument('--evaluate_seq_mats', type=int, default=None, help='Enable evaluate seq mats')
    parser.add_argument('--save_matches_video', type=int, default=None, help='Enable save matches video')

    parser.add_argument("--adaptive_bin", type=int, default=None, help="Enable adaptive binning")
    parser.add_argument('--max_odom', type=float, nargs=4, default=None, metavar=('max_ang_vel', 'max_lin_speed', 'max_ang_acc', 'max_lin_acc'), help='Four weights as floats (e.g., --max_odom 0.1 0.2 0.3 0.4)')
    parser.add_argument('--odom_weights', type=float, nargs=4, default=None, metavar=('w_ang_vel', 'w_lin_speed', 'w_ang_acc', 'w_lin_acc'), help='Four weights as floats (e.g., --odom_weights 0.1 0.2 0.3 0.4)')
    parser.add_argument('--max_bins', type=int, default=None, help='Max num bins for event reconstruction')
    parser.add_argument('--adaptive_bin_tag', type=str, default=None, help='tag for adaptive binned folder')
    parser.add_argument('--count_bin', type=int, default=None, help='enable bin with fixed count')
    parser.add_argument('--events_per_bin', type=int, default=None, help='number of events per bin for fixed count binning')
    parser.add_argument('--bin_tag', type=str, default=None, help='tag for adaptive binning')
    parser.add_argument('--use_exponential', type=bool, default=None, help='Use exponential binning if True, linear otherwise')

    return parser

def apply_defaults(args):
    if args.dataset_type is None:
        raise ValueError("You must specify a valid dataset_type (e.g., NSAVP, Brisbane)")

    if args.dataset_type not in default_config:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    print(f"Using dataset type: {args.dataset_type}")
    defaults = default_config[args.dataset_type]
    for key, default_val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_val)

    # Convert integer flags to booleans
    int_to_bool_keys = ['normalize_ref', 'normalize_qry', 'shift', 'random_downsample_bool', 'patch']
    for key in int_to_bool_keys:
        val = getattr(args, key, None)
        if val is not None:
            setattr(args, key, bool(val))

    return args
