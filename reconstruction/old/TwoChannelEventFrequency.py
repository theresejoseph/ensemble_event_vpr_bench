
from reconstruction.base_reconstructor import BaseReconstructor
import numpy as np
from utils.h5_utils import H5Handler
import time
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2 
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from numba import njit

# class EventReconstructor(BaseReconstructor):
#     def __init__(self):
#         pass
#     def reconstruct(self, eventsData=None, sensor_size=None, start_indices=None, end_indices=None):
#         """Your existing eventsEveryN frame creation logic"""
#         frame_times=[]
#         # print(numEvents)
#         count_array_3d = np.zeros((len(end_indices),  sensor_size[1], sensor_size[0]), dtype=np.float64)
#         array_3d = np.zeros((len(end_indices),  sensor_size[1], sensor_size[0]), dtype=np.float64)
#         for i, (start_idx, end_idx) in enumerate(tqdm(zip(start_indices, end_indices), total=len(start_indices), desc="Reconstructing")):    
#             frame_times.append((start_idx, end_idx))
#             x = eventsData['x'][start_idx:end_idx]
#             y = eventsData['y'][start_idx:end_idx]
#             p = np.where(eventsData['p'][start_idx:end_idx]>0, 1, -1)
#             # '''Storing array_3d'''
#             # np.add.at(count_array_3d, (i, y, x), p)


#             pixel_polarity_sequences = defaultdict(list)
#             # Collect polarity sequences per pixel
#             for xi, yi, pi in zip(x, y, p):
#                 pixel_polarity_sequences[(yi, xi)].append(pi)

#             # Initialize the frequency (transition) frame
#             transition_frame = np.zeros((sensor_size[1], sensor_size[0]), dtype=np.float64)

#             # Count transitions at each pixel
#             for (yi, xi), sequence in pixel_polarity_sequences.items():
#                 transitions = sum(1 for i in range(1, len(sequence)) if sequence[i] == sequence[i - 1])
#                 transition_frame[yi, xi] = transitions

#             # Store it in the 3D array
#             array_3d[i] = transition_frame
#             del pixel_polarity_sequences

#             '''VIZ'''
#             if i == 50:
#                 plt.imshow(np.tanh(array_3d[i]))
#                 plt.title(f'EventFreq: {len(x)}')
#                 plt.savefig('./tempFreqFrame.png')

                    

                
#         return array_3d, frame_times



@njit
def count_transitions(x, y, p, height, width):
    off_on_frame = np.zeros((height, width), dtype=np.float64)
    on_off_frame = np.zeros((height, width), dtype=np.float64)

    idxs = y * width + x
    sort_order = np.argsort(idxs)
    x = x[sort_order]
    y = y[sort_order]
    p = p[sort_order]
    idxs = idxs[sort_order]

    prev_idx = -1
    prev_pol = -1
    off_on = 0
    on_off = 0

    for i in range(len(p)):
        curr_idx = idxs[i]
        curr_pol = p[i]

        if curr_idx != prev_idx:
            # store results from previous pixel
            if prev_idx != -1:
                yi, xi = divmod(prev_idx, width)
                off_on_frame[yi, xi] = off_on
                on_off_frame[yi, xi] = on_off

            # reset for new pixel
            off_on = 0
            on_off = 0
            prev_pol = curr_pol
        else:
            if prev_pol == 0 and curr_pol == 1:
                off_on += 1
            elif prev_pol == 1 and curr_pol == 0:
                on_off += 1
            prev_pol = curr_pol

        prev_idx = curr_idx

    # flush last pixel
    if prev_idx != -1:
        yi, xi = divmod(prev_idx, width)
        off_on_frame[yi, xi] = off_on
        on_off_frame[yi, xi] = on_off

    return off_on_frame, on_off_frame


@njit
def decayed_transitions(x, y, t, p, height, width, current_time, decay_factor):
    off_on_frame = np.zeros((height, width), dtype=np.float64)
    on_off_frame = np.zeros((height, width), dtype=np.float64)

    idxs = y * width + x
    sort_order = np.argsort(idxs)
    x = x[sort_order]
    y = y[sort_order]
    p = p[sort_order]
    t = t[sort_order]
    idxs = idxs[sort_order]

    prev_idx = -1
    prev_pol = -1

    for i in range(len(p)):
        curr_idx = idxs[i]
        curr_pol = p[i]
        curr_time = t[i]

        if curr_idx != prev_idx:
            prev_pol = curr_pol
        else:
            decay = np.exp(-(current_time - curr_time) / decay_factor)
            yi, xi = divmod(curr_idx, width)
            if prev_pol == 0 and curr_pol == 1:
                off_on_frame[yi, xi] += decay
            elif prev_pol == 1 and curr_pol == 0:
                on_off_frame[yi, xi] += decay
            prev_pol = curr_pol

        prev_idx = curr_idx

    return off_on_frame, on_off_frame




class EventReconstructor(BaseReconstructor):
    def __init__(self):
        pass


    def reconstruct(self, eventsData=None, sensor_size=None, start_indices=None, end_indices=None):
        height, width = sensor_size[1], sensor_size[0]
        frame_times = []
        array_4d = np.zeros((len(start_indices), 2, height, width), dtype=np.float64)

        for i, (start_idx, end_idx) in enumerate(tqdm(zip(start_indices, end_indices), total=len(start_indices), desc="Reconstructing")):
            frame_times.append((start_idx, end_idx))

            x = eventsData['x'][start_idx:end_idx]
            y = eventsData['y'][start_idx:end_idx]
            p_raw = eventsData['p'][start_idx:end_idx]
            p = np.where(p_raw > 0, 1, 0)

            off_on_frame, on_off_frame = count_transitions(x, y, p, height, width)
            # t = eventsData['t'][start_idx:end_idx]
            # current_time = t[-1]
            # off_on_frame, on_off_frame = decayed_transitions(x, y, t, p, height, width, current_time, decay_factor=0.1)


            array_4d[i, 0] = off_on_frame
            array_4d[i, 1] = on_off_frame

            # Optional VIZ
            if i == 50:
                off_on_norm = np.tanh(off_on_frame)
                on_off_norm = np.tanh(on_off_frame)

                rgb_image = np.zeros((height, width, 3), dtype=np.float32)
                rgb_image[:, :, 1] = off_on_norm
                rgb_image[:, :, 0] = on_off_norm

                rgb_uint8 = (np.clip(rgb_image, 0.0, 1.0) * 255).astype(np.uint8)

                plt.figure(figsize=(6, 6))
                plt.imshow(rgb_uint8)
                plt.title('Polarity Transitions RGB: R=ON→OFF, G=OFF→ON')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('./tempPolarityTransRGB.png')
                plt.close()

                assert False

        return array_4d, frame_times
