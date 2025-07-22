import numpy as np
from pathlib import Path
from datasets.brisbane_events import BrisbaneEventDataset, Brisbane_RGB_Dataset
from datasets.nsvap import NSVAPDataset, NSAVP_RGB_Dataset
import importlib
from parser_config import get_parser, apply_defaults
import os
from glob import glob
import gc
import re
import shutil



def args_for_load_save():
    parser = get_parser()
    args = parser.parse_args()
    args = apply_defaults(args)

    # # Now you can use args with all defaults applied
    # for arg, value in vars(args).items():
    #     print(f"{arg}: {value}")
    args.sequences = ["night","morning", "sunrise", "sunset1", "sunset2", "daytime",
                        'R0_FA0', 'R0_FS0', 'R0_FN0', 'R0_RA0', 'R0_RS0', 'R0_RN0']

    # Initialize dataset and reconstructor based on the parameters.
    if args.dataset_type.lower() == 'nsavp':
        dataset = NSVAPDataset(args.dataset_path)
    elif args.dataset_type.lower() == 'brisbane':
        dataset = BrisbaneEventDataset(args.dataset_path)
    # elif args.dataset_type.lower() == 'nyc':
    #     dataset = NYCEventDataset(args.dataset_path)
    else:
        raise ValueError("Unsupported dataset type")


    # Dynamically construct the module path
    if args.reconstruct_method_name!= 'RGB_camera':
        module_path = f"reconstruction.{args.reconstruct_method_name}"
        reconstruction_module = importlib.import_module(module_path)
        reconstructor_class = getattr(reconstruction_module, "EventReconstructor")
        reconstructor = reconstructor_class() 
    elif args.reconstruct_method_name == 'RGB_camera' and args.dataset_type.lower() == 'nsavp':
        reconstructor=None
        dataset = NSAVP_RGB_Dataset(args.dataset_path)
    elif args.reconstruct_method_name == 'RGB_camera' and args.dataset_type.lower() == 'brisbane':
        reconstructor=None
        dataset = Brisbane_RGB_Dataset(args.dataset_path)

    return args, dataset, reconstructor


def make_paths(args, sequence_name):
    # save to work qvpr
    # work_dir =f'/work/qvpr/data/processed/{args.dataset_type}-Event-Reconstructions'
    additional_tag= 'Event' if args.dataset_type == 'Brisbane' else ''
    work_dir=f'../data/{args.dataset_type}{additional_tag}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # if args.reconstruct_method_name == 'RGB_camera':
    #     processed_path = None
    #     images_dir = Path(work_dir) / args.reconstruct_method_name / sequence_name
    #     video_filename = f"{sequence_name}_{args.reconstruct_method_name}.mp4"
    if args.count_bin ==1 and not args.adaptive_bin:
        bin_type = "fixed"
        subfolder = f"{bin_type}_countbins_{args.events_per_bin}"
        filename_base = f"{sequence_name}_{args.reconstruct_method_name}_{args.events_per_bin}{args.exp_tag}"
        processed_path = os.path.join(work_dir, "image_reconstructions", subfolder, args.reconstruct_method_name, f"{filename_base}.npz")
        images_dir = Path(work_dir) / "image_reconstructions"/ subfolder/ args.reconstruct_method_name / sequence_name
        video_filename = f"{filename_base}.mp4"
    elif args.adaptive_bin == 1:
        bin_param = f"minres_{args.min_time_res}_maxres{round(args.max_bins*args.min_time_res,2)}"
        subfolder = f"{args.adaptive_bin_tag}_timebins_{args.max_bins}"
        filename_base = f"{sequence_name}_{args.reconstruct_method_name}_{bin_param}{args.exp_tag}"
        processed_path = os.path.join(work_dir, "image_reconstructions", subfolder, args.reconstruct_method_name, f"{filename_base}.npz")
        images_dir = Path(work_dir) / "image_reconstructions"/ subfolder/ args.reconstruct_method_name / sequence_name
        video_filename = f"{filename_base}.mp4"
    else:
        bin_type = "fixed"
        subfolder = f"{bin_type}_timebins_{args.time_res}"
        filename_base = f"{sequence_name}_{args.reconstruct_method_name}_{args.time_res}{args.exp_tag}"
        processed_path = os.path.join(work_dir, "image_reconstructions", subfolder, args.reconstruct_method_name, f"{filename_base}.npz")
        images_dir = Path(work_dir) / "image_reconstructions"/ subfolder/ args.reconstruct_method_name / sequence_name
        video_filename = f"{filename_base}.mp4"

    
    # Assign to args
    args.processed_path = processed_path
    args.save_images_dir = images_dir
    args.video_filename = video_filename
    args.subfolder_dir = os.path.join(work_dir, "image_reconstructions", subfolder)



def args_for_load_save(reconstruct_method_name):
    parser = get_parser()
    args = parser.parse_args([])
    args.dataset_type = dataset_type
    args = apply_defaults(args)
        
    # Override parsed args with specific values
    args.adaptive_bin = adaptive_bin
    args.save_frames_video = 0
    args.sequences = ["night","morning", "sunrise", "sunset1", "sunset2", "daytime",
                        'R0_FA0', 'R0_FS0', 'R0_FN0', 'R0_RA0', 'R0_RS0', 'R0_RN0']

    args.adaptive_bin_tag = bin_tag
    args.reconstruct_method_name = reconstruct_method_name
    args.time_res = time_res
    args.count_bin = count_bin  # 1 for event count binning, 0 for time binning
    args.events_per_bin = events_per_bin  # Number of events per bin for eventCount reconstruction


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

    return args, dataset, reconstructor



def args_for_vpr(idR, idQ, reconstruct_method_name):
    args_vpr = parse.parse_arguments(method)
    # Override parsed args
    args_vpr.idR = idR
    args_vpr.idQ = idQ
    args_vpr.reconstruct_method_name = reconstruct_method_name
    args_vpr.adaptive_bin = adaptive_bin
    args_vpr.sequences = ["night","morning", "sunrise", "sunset1", "sunset2", "daytime",
                        'R0_FA0', 'R0_FS0', 'R0_FN0', 'R0_RA0', 'R0_RS0', 'R0_RN0']

    args_vpr.saveSimMat = True
    args_vpr.expTag = ''
    args_vpr.adaptive_bin_tag = bin_tag 
    args_vpr.time_res = time_res
    args_vpr.count_bin = count_bin  # 1 for event count binning, 0 for time binning
    args_vpr.events_per_bin = events_per_bin  # Number of events per bin for eventCount reconstruction
    args_vpr.patch_or_frame = patch_or_frame
    args_vpr.seq_len = args_cli.seq_len  # Sequence length for VPR
    
    return args_vpr

