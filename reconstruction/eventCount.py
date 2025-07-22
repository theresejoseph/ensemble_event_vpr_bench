from reconstruction.base_reconstructor import BaseReconstructor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class EventReconstructor(BaseReconstructor):
    def __init__(self):
        pass

    def reconstruct(self, eventsData, sensor_size, start_indices, end_indices, hp_loc=None):
        """
        Args:
            events: Structured numpy array with fields 'x', 'y', 'p' (polarity).
                    Polarity should be in {0, 1}.
            sensor_size: (width, height) of the sensor.
            start_indices: Frame window start indices.
            end_indices: Frame window end indices.
        Returns:
            histograms: np.array of shape (num_frames, 2, height, width), polarity-separated.
            frame_times: list of (start_idx, end_idx) tuples.
        """
        height, width = sensor_size[1], sensor_size[0]
        frame_times = []
        reconstructed_images = np.zeros((len(start_indices), height, width, 3), dtype=np.uint8)


        for i, (start_idx, end_idx) in enumerate(tqdm(zip(start_indices, end_indices), total=len(start_indices), desc="Polarity Binning")):
            frame_times.append((start_idx, end_idx))

            x = eventsData['x'][start_idx:end_idx]
            y = eventsData['y'][start_idx:end_idx]
            p = eventsData['p'][start_idx:end_idx]

            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x = x[valid].astype(np.int32)
            y = y[valid].astype(np.int32)
            p = p[valid].astype(np.int32)

            hist = np.zeros((2, height, width), dtype=np.float32)
            np.add.at(hist, (p, y, x), 1)

            # Apply tanh to compress dynamic range
            off_channel = abs(np.tanh(hist[0]))
            on_channel = abs(np.tanh(hist[1]))

            # Per-channel normalization (if max > 0) → same as original behavior
            if off_channel.max() > 0:
                off_img = np.clip(off_channel / (off_channel.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
            else:
                off_img = np.zeros_like(off_channel, dtype=np.uint8)

            if on_channel.max() > 0:
                on_img = np.clip(on_channel / (on_channel.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
            else:
                on_img = np.zeros_like(on_channel, dtype=np.uint8)

            # Compose RGB image (G = ON, R = OFF)
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = off_img   # Red = OFF
            rgb_image[:, :, 1] = on_img    # Green = ON
            # Blue remains 0 → background is black

            reconstructed_images[i] = rgb_image

            if i == 0:
                rgb_uint8 = (np.clip(rgb_image, 0.0, 1.0) * 255).astype(np.uint8)
                plt.figure(figsize=(6, 6))
                plt.imshow(rgb_uint8)
                plt.title('Polarity Binned Histogram - Frame {i}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('./plots/tempPolarityHistogram.png')
                plt.close()

                # assert False, "Stopping after 50 frames for debugging"


        return reconstructed_images, frame_times

