import numpy as np
from pathlib import Path
# from datasets.nyc_events import NYCEventDataset
from datasets.brisbane_events import BrisbaneEventDataset, Brisbane_RGB_Dataset
from datasets.nsvap import NSVAPDataset, NSAVP_RGB_Dataset
import importlib
from parser_config import get_parser, apply_defaults
np.random.seed(1)
import cv2
import os
from glob import glob
import gc
import re
import shutil
from pathlib import Path
import re


def args_for_load_save():
    parser = get_parser()
    args = parser.parse_args()
    args = apply_defaults(args)

    args.sequences = ["night","morning", "sunrise", "sunset1", "sunset2", "daytime",
                        'R0_FA0', 'R0_FS0', 'R0_FN0', 'R0_RA0', 'R0_RS0', 'R0_RN0']

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



def extract_frame_index(path):
    # Match pattern like ...@frame_42@.jpg
    match = re.search(rf"@{re.escape('frame')}_([0-9]+)@", path.name)
    return int(match.group(1)) if match else float('inf')  # sort non-matching files last



def save_video(sequence_dir, video_filename=None):
    video_path = Path(sequence_dir).parent / video_filename
    if not video_path.exists():
        print(f"Saving video to {video_path}")
        image_paths = sorted(Path(sequence_dir).glob("*.jpg"), key=extract_frame_index)

        if not image_paths:
            print("Warning: No images found to write video.")
            return

        # Read first image to get size
        sample_image = cv2.imread(str(image_paths[0]))
        if sample_image is None:
            print(f"Error: Failed to read image {image_paths[0]}")
            return
        height, width = sample_image.shape[:2]

        # Initialize video writer
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))

        # Write each frame
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Could not read {img_path}, skipping.")
                continue
            video_writer.write(frame)
        video_writer.release()
        print(f"Video successfully saved to {video_path}")



def load_save_data(dataset, reconstructor, args, ref_or_qry, return_data=True):
    '''Processing reference frames and positions'''
    seq_idx = args.ref_seq_idx if ref_or_qry == 'ref' else args.qry_seq_idx
    sequence_name = args.sequences[seq_idx]
    
    make_paths(args, sequence_name)
    print(f"Processing sequence: {sequence_name} for {ref_or_qry}  from {args.save_images_dir }", flush=True)
    # Only process and save frames if the directory does not exist or is empty
    sequence_dir = args.save_images_dir
    if not os.path.exists(sequence_dir):
        frames, gt_positions = dataset.process_sequence(args, ref_or_qry, reconstructor)
        os.makedirs(sequence_dir, exist_ok=True)
        print(f"Saving {len(frames)} images into {sequence_dir}")
        saved_count, skipped_count = 0, 0
        for i, (frame, position) in enumerate(zip(frames, gt_positions)):
            utm_east, utm_north = position[0], position[1]
            filename = f"@{utm_east:.6f}@{utm_north:.6f}@frame_{i}@.jpg"
            filepath = os.path.join(sequence_dir, filename)

            if os.path.exists(filepath):
                skipped_count += 1
                continue

            if frame.ndim == 2:  # Grayscale
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3:  # RGB
                if frame.dtype != np.uint8:
                    max_val = frame.max()
                    frame = frame / max_val if max_val > 0 else frame
                    frame = (frame * 255).astype(np.uint8)
                rgb_image = frame # RGB → BGR for OpenCV
            else:
                print(f"Skipping unsupported frame shape {frame.shape} at index {i}")
                skipped_count += 1
                continue

            cv2.imwrite(filepath, rgb_image)
            saved_count += 1

        print(f"Saved {saved_count} new images, skipped {skipped_count}.")

        # --- Save video ---
        if args.save_frames_video:
            save_video(sequence_dir, args.video_filename)
    elif return_data:
        sequence_dir = args.save_images_dir
        image_paths = sorted(list(Path(sequence_dir).glob("*.jpg")) + list(Path(sequence_dir).glob("*.png")), key=extract_frame_index)
        frames, gt_positions = [], []
        for img_path in image_paths:
            parts = img_path.name.split('@')
            if len(parts) >= 4:
                try:
                    utm_east = float(parts[1])
                    utm_north = float(parts[2])
                    position = np.array([utm_east, utm_north])
                    gt_positions.append(position)

                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        print(f"Warning: Failed to load frame {img_path}, skipping.")
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Back to RGB
                    frames.append(frame)
                except ValueError:
                    print(f"Warning: Failed to parse position in filename {img_path.name}, skipping.")
            else:
                print(f"Skipping improperly formatted filename: {img_path.name}")

        frames = np.stack(frames) if frames else np.empty((0,))
        gt_positions = np.stack(gt_positions) if gt_positions else np.empty((0, 2))
        print(f"Loaded {len(frames)} frames and positions from {sequence_dir}")
    else:
        frames, gt_positions = [], []

        
    return frames,  gt_positions



if __name__ == "__main__":

    args, dataset, reconstructor = args_for_load_save()

    args.save_frames_video = True
    print(args.ref_seq_idx, args.qry_seq_idx, flush=True)
    ref_frames,ref_gt_positions = load_save_data(dataset, reconstructor, args, 'ref', return_data= False)
    qry_frames, qry_gt_positions  = load_save_data(dataset, reconstructor, args, 'qry',return_data= False)

        