from load_and_save import load_save_data
from vpr_methods_evaluation.main import run_vpr_save_results, run_vpr, run_vpr_fill_auc
from vpr_methods_evaluation import parse
from parser_config import get_parser, apply_defaults
from datasets.brisbane_events import BrisbaneEventDataset, Brisbane_RGB_Dataset
from datasets.nsvap import NSVAPDataset, NSAVP_RGB_Dataset
import importlib
import argparse

def default_args_testing():
    parser_train = argparse.ArgumentParser()

    parser_train.add_argument("--method", type=str, default="mixvpr", help="vpr method to use")
    parser_train.add_argument("--dataset_type", type=str, default="Brisbane", help="dataset type (e.g., Brisbane, NSAVP)")
    
    parser_train.add_argument("--reconstruct_method_name", type=str, default="timeSurface", help="Reconstruction method name (e.g., eventCount, timeSurface, e2vid)")
    parser_train.add_argument("--ref_seq_idx", type=int, default=6, help="Reference sequence index")
    parser_train.add_argument("--qry_seq_idx", type=int, default=7, help="Query sequence index")
    parser_train.add_argument("--seq_len", type=int, default=1, help="Sequence length for VPR")
    parser_train.add_argument("--patch_or_frame", type=str, default="frame", help="Use 'patch' for patch-based, 'frame' for frame-based reconstruction")
    parser_train.add_argument("--patch_num_cols", type=int, default="2", help="Num patch columns for patch-based reconstruction")
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
    args_cli = parser_train.parse_args()

    args_cli.sequences=["night","morning", "sunrise", "sunset1", "sunset2", "daytime",
     'R0_FA0', 'R0_FS0', 'R0_FN0', 'R0_RA0', 'R0_RS0', 'R0_RN0', "night_training",
     "morning_training", "sunrise_training", "sunset1_training", "sunset2_training", "daytime_training",]
    args_cli.dataset_type = 'NSAVP' if args_cli.ref_seq_idx >= 6 or args_cli.qry_seq_idx >= 6 else 'Brisbane'
    args_cli.dataset_type = 'Brisbane' if args_cli.ref_seq_idx >= 12 or args_cli.qry_seq_idx >= 12 else args_cli.dataset_type
    
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
    
    print(args.dataset_type)

    return args, dataset, reconstructor




def args_for_vpr(args_cli, reconstruct_method_name, idR, idQ):
    args_vpr = parse.parse_arguments(args_cli.method)
    # Override parsed args
    args_vpr.idR = idR
    args_vpr.idQ = idQ
    args_vpr.sequences = args_cli.sequences
    args_vpr.dataset_type = args_cli.dataset_type
    args_vpr.reconstruct_method_name = reconstruct_method_name
    args_vpr.saveSimMat = True #<---------------------------------------------------- change this to True if you want to save the similarity matrix
    
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
    args_ls, dataset, reconstructor = args_for_load_save(args_cli, reconstruct_method_name)
    args_ls.ref_seq_idx = idR
    args_ls.qry_seq_idx = idQ

    frames_r, _ = load_save_data(dataset, reconstructor, args_ls, 'ref', return_data=False)
    frames_q, _ = load_save_data(dataset, reconstructor, args_ls, 'qry', return_data=False)

    return reconstruct_method_name, idR, idQ, [len(frames_r), len(frames_q)]



def run_single_experiment():
    """
    Run the place recognition experiment with the given arguments.
    """
    recalls=[]
    seq_id_to_name = {0:'night', 1:'morning', 2: 'sunrise', 3:'sunset1', 4:'sunset2', 5:'daytime', 
                  6: 'R0_FA0', 7: 'R0_FS0', 8: 'R0_FN0', 9: 'R0_RA0', 10: 'R0_RS0', 11: 'R0_RN0'}

    # experiment_pairs = [ (3,0),(3, 1), (3, 2),(3, 4), (3, 5)] if args_cli.dataset_type == 'Brisbane' else [(6, 7),(6, 8), (8, 7), (9,11), (10, 11)]

    idR = args_cli.ref_seq_idx
    idQ = args_cli.qry_seq_idx  
    reconstruct_method_name = args_cli.reconstruct_method_name
    print(f"Running VPR for idR={idR}, idQ={idQ}, reconstruct_method_name={reconstruct_method_name}")
    # if args_cli.count_bin == 1 and args_cli.events_per_bin == 1_000_000:
    #     print("Using eventCount reconstruction with 1M events per bin")
    #     process_pair(args_cli, reconstruct_method_name, idR, idQ)
    args_vpr = args_for_vpr(args_cli, reconstruct_method_name, idR, idQ)
    recall_at_1 = run_vpr_save_results(args_vpr)
    
    print(f"Recall at 1: {recall_at_1}")


from tqdm import tqdm

def run_all_experiments():
    """
    Run the place recognition experiment with the given arguments.
    """
    experiment_pairs = [(3, 0), (3, 1), (3, 2), (3, 4), (3, 5),
                        (6, 7), (6, 8), (8, 7), (9, 11), (10, 11)]
    recons = ['eventCount', 'eventCount_noPolarity', 'timeSurface', 'e2vid']
    methods = ['mixvpr', 'megaloc', 'netvlad', 'cosplace']
    time_res_list = [1.0, 0.5, 0.25, 0.2, 0.15, 0.1]

    print("=== Running TIME-BASED binning experiments ===")
    for seq_len in [20, 10, 1]:
        args_cli.seq_len = seq_len
        print(f"\n▶ Sequence length: {seq_len}")

        args_cli.patch_or_frame = 'frame'

        for time_res in time_res_list:
            args_cli.time_res = time_res
            args_cli.count_bin = 0
            print(f"    ▶ Time resolution: {time_res:.3f}s")

            for reconstruct_method_name in recons:
                args_cli.reconstruct_method_name = reconstruct_method_name
                print(f"      ▶ Reconstruction: {reconstruct_method_name}")

                for method in methods:
                    args_cli.method = method
                    print(f"        ▶ Method: {method}")

                    for idR, idQ in tqdm(experiment_pairs, desc="          ▶ Experiment Pairs"):
                        args_cli.ref_seq_idx = idR
                        args_cli.qry_seq_idx = idQ

                        print(f"            → Running VPR | R={idR} Q={idQ} | method={method}, recon={reconstruct_method_name}, Δt={time_res:.3f}s, len={seq_len}")
                        args_vpr = args_for_vpr(args_cli, reconstruct_method_name, idR, idQ)
                        run_vpr_fill_auc(args_vpr)

    print("\n=== Running COUNT-BASED binning experiments ===")
    events_per_bins = [100_000, 200_000, 300_000, 500_000, 1_000_000]
    for seq_len in [1]:
        args_cli.seq_len = seq_len
        print(f"\n▶ Sequence length: {seq_len}")
        args_cli.patch_or_frame = 'frame'

        for events_per_bin in events_per_bins:
            args_cli.events_per_bin = events_per_bin
            args_cli.count_bin = 1
            print(f"    ▶ Events per bin: {events_per_bin:,}")

            for reconstruct_method_name in recons:
                args_cli.reconstruct_method_name = reconstruct_method_name
                print(f"      ▶ Reconstruction: {reconstruct_method_name}")

                for method in methods:
                    args_cli.method = method
                    print(f"        ▶ Method: {method}")

                    for idR, idQ in tqdm(experiment_pairs, desc="          ▶ Experiment Pairs"):
                        args_cli.ref_seq_idx = idR
                        args_cli.qry_seq_idx = idQ

                        print(f"            → Running VPR | R={idR} Q={idQ} | method={method}, recon={reconstruct_method_name}, events/bin={events_per_bin}, len={seq_len}")
                        args_vpr = args_for_vpr(args_cli, reconstruct_method_name, idR, idQ)
                        run_vpr_fill_auc(args_vpr)


if __name__ == "__main__":
    args_cli = default_args_testing()

    # run_single_experiment()
    # run_all_experiments()
    run_single_experiment()

    



