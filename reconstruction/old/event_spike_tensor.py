import numpy as np
import h5py
from reconstruction.base_reconstructor import BaseReconstructor
from utils.h5_utils import H5Handler
import time
from scipy import signal
import matplotlib.pyplot as plt
import time
from scipy.signal import fftconvolve

class EST(BaseReconstructor):
    def __init__(self):
        self.methodName = 'EventSpikeTensor_Conv'  

    def reconstruct(self, events_path, time_windows, decay_factor=None):
        # time_windows=time_windows[::5]
        with h5py.File(events_path, 'r') as f:
            frame_width = f["events"].attrs["width"]
            frame_height = f["events"].attrs["height"]
            
            frame_times,times = [], []
            all_timestamps = f["events/timestamps"]
            
            
            # fig,(ax1,ax2, ax3)=plt.subplots(1,3)
            kernelT = np.arange(0, 50, 1)
            k = 0.1
            kernel = 2*np.exp(-k * kernelT)
            spike_tensor = np.zeros((len(time_windows), frame_height, frame_width), dtype=np.float64)
            
            for i, (start_time, end_time) in enumerate(time_windows):
                t0=time.time()
                # Find event indices for the current time window
                start_idx = H5Handler.binary_search_chunks(all_timestamps, start_time)
                end_idx = H5Handler.binary_search_chunks(all_timestamps, end_time)
                timeIncrm=int((end_time-start_time)/1e6)
                print(start_time,end_time,timeIncrm) 

                
                
                x = f["events/x_coordinates"][start_idx:end_idx]
                y = f["events/y_coordinates"][start_idx:end_idx]
                p = f["events/polarities"][start_idx:end_idx].astype(np.int16)
                p[p==0]=-1
                window_timestamps = all_timestamps[start_idx:end_idx]
                t = [int((window_t -start_time)/1e6) for window_t in window_timestamps]
                timeLen=len(np.unique(t))
                array_3d = np.zeros((timeLen, frame_height, frame_width), dtype=np.float64)
                np.add.at(array_3d, (t, y, x), p)

                output_array = np.zeros_like(array_3d)
                for y in range(array_3d.shape[1]):
                    for x in range(array_3d.shape[2]):
                        output_array[:, y, x] = fftconvolve(array_3d[:, y, x], kernel, mode='full')[:timeLen]
               
            
                spike_tensor[i]=output_array[-1,:,:]
                

                runtime = time.time() - t0
                times.append(runtime)
                approx_remTime = ((len(time_windows) - i) * np.median(times)) / 60
                print(i, start_idx, end_idx, round(runtime, 2), 'secs', round(approx_remTime, 1), 'approx. mins left')
           
            return spike_tensor, time_windows
        



    # def reconstruct(self, events_path, time_windows, decay_factor=0.1, num_bins=100):
    #     # Open the HDF5 file and load event metadata and data
    #     with h5py.File(events_path, 'r') as f:
    #         frame_width = f["events"].attrs["width"]
    #         frame_height = f["events"].attrs["height"]
    #         # Assume the file contains datasets for 'timestamps', 'x', and 'y'
    #         all_timestamps = f["events/timestamps"]

    #         # Initialize the spike tensor (one slice per time window)
    #         spike_tensor = np.zeros((len(time_windows), frame_height, frame_width), dtype=np.float64)

    #         # Create a 1D exponential kernel over a normalized time axis
    #         t_kernel = np.linspace(0, 1, num_bins)
    #         kernel = np.exp(-t_kernel / decay_factor)

    #         frame_times,times=[],[]
    #         # Process each time window
    #         for i, (start_time, end_time) in enumerate(time_windows):
    #             t=time.time()
    #             # Find event indices for the current time window
    #             start_idx = H5Handler.binary_search_chunks(all_timestamps, start_time)
    #             end_idx = H5Handler.binary_search_chunks(all_timestamps, end_time)
    #             frame_times.append((start_idx, end_idx))

    #             window_x = f["events/x_coordinates"][start_idx:end_idx]
    #             window_y = f["events/y_coordinates"][start_idx:end_idx]
    #             window_p = f["events/polarities"][start_idx:end_idx]
    #             window_timestamps = all_timestamps[start_idx:end_idx]

    #             # Initialize an accumulator for this window (a frame)
    #             accumulator = np.zeros((frame_height, frame_width), dtype=np.float64)
                
    #             # Process each event asynchronously
    #             max_dt = 10.0 
    #             for j in range(len(window_timestamps)):
    #                 # Delta between the window end and the event time
    #                 dt = end_time - window_timestamps[j]
    #                 # Ensure dt is non-negative
    #                 if dt < 0:
    #                     dt = 0.0
    #                 # Clip dt to avoid too large values
    #                 dt = min(dt, max_dt)
    #                 # Compute the exponential weight (the kernel value)
    #                 weight = np.exp(-dt / decay_factor)
    #                 # Use the polarity to determine sign
    #                 value = weight if window_p[j] > 0 else -weight
    #                 # Accumulate at the event pixel location
    #                 accumulator[window_y[j], window_x[j]] += value
                
    #             spike_tensor[i]=accumulator

    #             runtime = time.time() - t
    #             times.append(runtime)
    #             approx_remTime = ((len(time_windows) - i) * np.median(times)) / 60
    #             print(i, start_idx, end_idx, round(runtime, 2), 'secs', round(approx_remTime, 1), 'approx. mins left')
    #             # plt.imshow(accumulator)
    #             # plt.show()
    #             # assert False
    #         return spike_tensor, frame_times
