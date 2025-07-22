
from reconstruction.base_reconstructor import BaseReconstructor
import numpy as np
from utils.h5_utils import H5Handler
import matplotlib.pyplot as plt
from tqdm import tqdm
from reconstruction.base_reconstructor import BaseReconstructor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class EventReconstructor(BaseReconstructor):
    def __init__(self):
        pass

    def reconstruct(self, eventsData, sensor_size, start_indices, end_indices, hp_loc=None):
        frame_times = []
        height, width = sensor_size[1], sensor_size[0]
        reconstructed_images = np.zeros((len(end_indices), height, width), dtype=np.uint8)  # RGB grayscale image

        for i, (start_idx, end_idx) in enumerate(tqdm(zip(start_indices, end_indices), total=len(start_indices), desc="Reconstructing")):
            frame_times.append((start_idx, end_idx))
            x = eventsData['x'][start_idx:end_idx]
            y = eventsData['y'][start_idx:end_idx]

            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x = x[valid].astype(np.int32)
            y = y[valid].astype(np.int32)

            count_map = np.zeros((height, width), dtype=np.float32)
            np.add.at(count_map, (y, x), 1)

            # Normalize using tanh and map to uint8 grayscale RGB
            norm_map = np.tanh(count_map)  # Optional: could use other norms (e.g. /max)
            norm_map = (norm_map + 1.0) / 2.0  # Rescale from [-1, 1] → [0, 1]
            gray_uint8 = (norm_map * 255).astype(np.uint8)

            # rgb_image = np.stack([gray_uint8]*3, axis=-1)  # grayscale → RGB
            reconstructed_images[i] = gray_uint8

            if i == 50:
                plt.imshow(gray_uint8)
                plt.title(f'EventUnsignedCount RGB: {len(x)} events')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('./plots/tempUnsignedFrame.png')
                plt.close()

        return reconstructed_images, frame_times
