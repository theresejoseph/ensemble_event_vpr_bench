import numpy as np
from reconstruction.base_reconstructor import BaseReconstructor
import matplotlib.pyplot as plt
from tqdm import tqdm

class EventReconstructor(BaseReconstructor):
    def __init__(self):
        pass

    def reconstruct(self, eventsData, sensor_size, start_indices, end_indices, decay_factor=0.1, hp_loc=None):
        """
        Constructs RGB uint8 frames from event data using tonic-style exponential decay time surfaces.

        Parameters:
        - eventsData (np.ndarray): Structured array with fields 'x', 'y', 't', 'p'
        - sensor_size (tuple): (width, height)
        - start_indices (list[int])
        - end_indices (list[int])
        - decay_factor (float): Time decay factor τ

        Returns:
        - reconstructed_images: shape (N, H, W, 3), RGB images in uint8
        - frame_times: list of (start_idx, end_idx)
        """
        height, width = sensor_size[1], sensor_size[0]
        reconstructed_images = np.zeros((len(start_indices), height, width, 3), dtype=np.uint8)
        frame_times = []

        for i, (start_idx, end_idx) in enumerate(tqdm(zip(start_indices, end_indices), total=len(start_indices), desc="Reconstructing")):
            frame_times.append((start_idx, end_idx))
            memory = np.ones((2, height, width), dtype=np.float64) * -np.inf  # memory for [ON, OFF]

            x = eventsData['x'][start_idx:end_idx]
            y = eventsData['y'][start_idx:end_idx]
            t = eventsData['t'][start_idx:end_idx]
            p = eventsData['p'][start_idx:end_idx]

            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & ((p == 0) | (p == 1))
            x, y, t, p = x[valid], y[valid], t[valid], p[valid]

            if len(x) == 0:
                continue

            indices = (p, y, x)
            memory[indices] = t

            diff = -((t[-1] + decay_factor) - memory)
            surface = np.exp(diff / decay_factor)
            surface = np.clip(surface, 0.0, 1.0)

            # Apply tanh normalization
            on_channel = abs(np.tanh(surface[1]))  # ON → Red
            off_channel = abs(np.tanh(surface[0]))  # OFF → Green

            # Normalize each channel independently to 0–255 range if nonzero
            if on_channel.max() > 0:
                on_img = np.clip(on_channel / (on_channel.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
            else:
                on_img = np.zeros_like(on_channel, dtype=np.uint8)

            if off_channel.max() > 0:
                off_img = np.clip(off_channel / (off_channel.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
            else:
                off_img = np.zeros_like(off_channel, dtype=np.uint8)

            # Compose RGB image (R=ON, G=OFF)
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = on_img   # Red = ON
            rgb_image[:, :, 1] = off_img  # Green = OFF
            # Blue remains 0 → background black

            reconstructed_images[i] = rgb_image

            if i == 50:
                plt.figure(figsize=(6, 6))
                plt.imshow(rgb_image)
                plt.title('Polarity TimeSurface RGB: R=ON, G=OFF')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('./plots/tempTimeSurfaceRGB.png')
                plt.close()

        return reconstructed_images, frame_times
