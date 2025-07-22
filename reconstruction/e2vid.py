
# import sys
# sys.path.append('./reconstruction_repos/')

from reconstruction.base_reconstructor import BaseReconstructor

class EventReconstructor(BaseReconstructor):
    def __init__(self):
        pass


    def reconstruct(self, eventsData, sensor_size, start_indices, end_indices, hp_loc=None):
        from e2vid.utils.loading_utils import load_model, get_device
        import numpy as np
        from e2vid.utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
        from e2vid.image_reconstructor import ImageReconstructor
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        from argparse import Namespace
        import torch

        args = Namespace(
            path_to_model='./e2vid/pretrained/E2VID_lightweight.pth.tar',
            use_gpu=True,
            fixed_duration=False,
            window_size=None,
            window_duration=33.33,
            num_events_per_pixel=0.35,
            skipevents=0,
            suboffset=0,
            compute_voxel_grid_on_cpu=False,
            # Options for inference
            display=False,
            show_events=False,
            event_display_mode='default',
            num_bins_to_show=5,
            display_border_crop=0,
            display_wait_time=1,
            hot_pixels_file=hp_loc,
            unsharp_mask_amount=0,
            unsharp_mask_sigma=0,
            bilateral_filter_sigma=0,
            flip=False,
            Imin=0,
            Imax=1,
            auto_hdr=False,
            auto_hdr_median_filter_size=0,
            color=False,
            no_normalize=False,
            no_recurrent=False,
            output_folder="./",
            dataset_name="bris_intensity_images",
        )

        width, height = sensor_size[0],sensor_size[1]
        print('Sensor size: {} x {}'.format(width, height))
        model = load_model(args.path_to_model)
        device = get_device(args.use_gpu)

        model = model.to(device)
        model.eval()

        

        if args.compute_voxel_grid_on_cpu:
            print('Will compute voxel grid on CPU.')
        reconstructed_images = []
        height, width = sensor_size[1], sensor_size[0]
        frame_times = []

        for i, (start_idx, end_idx) in enumerate(tqdm(zip(start_indices, end_indices), desc="Intensity Reconstruction", total=len(start_indices))):
            frame_times.append((start_idx, end_idx))
            im_recon = ImageReconstructor(model, height, width, model.num_bins, args)
            # Handle case when there are no events in this window
            if start_idx >= end_idx:
                print(f"Warning: No events in window {i} (start_idx={start_idx}, end_idx={end_idx})")
                # Create a zero image for this frame
                if hasattr(model, 'num_bins'):
                    zero_image = np.zeros((height, width), dtype=np.float32)  # Or whatever shape your model expects
                else:
                    zero_image = np.zeros((height, width), dtype=np.float32)
                reconstructed_images.append(zero_image)
                continue

            # Stack into (N, 4): [t,x, y,  p]
            ev = np.stack([
                eventsData['t'][start_idx:end_idx],  # t should be third
                eventsData['x'][start_idx:end_idx],  # x should be first, not t 
                eventsData['y'][start_idx:end_idx],
                # Convert -1/+1 polarities to 0/1
                eventsData['p'][start_idx:end_idx] > 0,  # Ensure polarity is 0/1
            ], axis=1)

            # Step 1: Apply valid coordinate filtering
            valid = (ev[:, 1] >= 0) & (ev[:, 1] < width) & (ev[:, 2] >= 0) & (ev[:, 2] < height)
            event_window_np = ev[valid]
            
            # Handle case where all events are filtered out (invalid coordinates)
            if event_window_np.shape[0] == 0:
                print(f"Warning: All events filtered out in window {i} (invalid coordinates)")
                # Create a zero image for this frame
                if hasattr(model, 'num_bins'):
                    zero_image = np.zeros((height, width), dtype=np.float32)
                else:
                    zero_image = np.zeros((height, width), dtype=np.float32)
                reconstructed_images.append(zero_image)
                continue

            last_timestamp = event_window_np[-1, 0]
            if args.compute_voxel_grid_on_cpu:
                event_tensor = events_to_voxel_grid(event_window_np,
                                                    num_bins=model.num_bins,
                                                    width=width,
                                                    height=height)
                event_tensor = torch.from_numpy(event_tensor)
            else:
                event_tensor = events_to_voxel_grid_pytorch(event_window_np,
                                                            num_bins=model.num_bins,
                                                            width=width,
                                                            height=height,
                                                            device=device)

            num_events_in_window = event_window_np.shape[0]
            image = im_recon.update_reconstruction(event_tensor, start_idx + num_events_in_window, last_timestamp)
            image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            reconstructed_images.append(image)
            
            if i == 50:
                # Debugging: Save the 50th frame as an image
                plt.figure(figsize=(10, 6))
                plt.imshow(image, cmap='gray')
                plt.title("E2VID Reconstruction - Frame 50")
                plt.colorbar()
                plt.savefig("./plots/tempE2VID_frame_50.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Debug: Frame 50 saved.")
                # Remove the assert to continue processing
                # assert False, "Stopping after 50 frames for debugging. Check tempE2VID_frame_50.png"
                        
        return np.array(reconstructed_images), frame_times