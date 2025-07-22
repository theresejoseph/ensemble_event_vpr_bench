import h5py
import numpy as np
import os 


def save_processed_sequence(path, frames, gt_positions, frame_times, frame_spacing_meters, frame_duration, endIdx):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        frames=frames,
        gt_positions=gt_positions,
        frame_times=np.array(frame_times),
        frame_spacing_meters=frame_spacing_meters,
        frame_duration=frame_duration,
        endIdx=endIdx
    )


def load_processed_sequence(path, startIdx=0, endIdx=3400, incr=1, gt_pos_only=False):
    data = np.load(path, allow_pickle=True)

    if gt_pos_only:
        return data["gt_positions"][startIdx:endIdx:incr]
    
    frames = data['frames'][startIdx:endIdx:incr]
    gt_positions = data['gt_positions'][startIdx:endIdx:incr]
    return frames, gt_positions

class H5Handler:
    @staticmethod
    def save_processed_sequence(filepath, frames, positions, time_windows, frame_spacing_meters, frame_duration, endIdx, metadata=None):
        with h5py.File(filepath, 'w') as f:
            # Create groups
            frames_group = f.create_group('frames')
            gt_group = f.create_group('ground_truth')
            time_group = f.create_group('frame_times')
            
            # Save data with compression
            frames_group.create_dataset('event_frames', data=frames, 
                                     compression='gzip', compression_opts=9)
            gt_group.create_dataset('positions', data=np.array(positions), 
                                  compression='gzip', compression_opts=9)
            
            # Save time windows
            starts, ends = zip(*time_windows)
            time_group.create_dataset('start_indices', data=np.array(starts))
            time_group.create_dataset('end_indices', data=np.array(ends))
            
            # Add metadata
            f.attrs.update({
                'frame_spacing_meters': frame_spacing_meters,
                'frame_duration': frame_duration, #nanoseconds 
                'endIdx': endIdx
            })
            if metadata:
                f.attrs.update(metadata)
    

    @staticmethod
    def load__processed_sequence(filepath, startIdx=None, endIdx=None, incr=1, gt_pos_only= False):
        
        with h5py.File(filepath, 'r') as f:
            # Load frames
            
            if endIdx==None:
                endIdx=len(f['frames/event_frames'])
                
            if gt_pos_only != True:
                # Load ground truth
                if endIdx>0:
                    frames = f['frames/event_frames'][startIdx:endIdx:incr]
                    gt_positions = f['ground_truth/positions'][startIdx:endIdx:incr]
                else:
                    frames = f['frames/event_frames'][endIdx:]
                    gt_positions = f['ground_truth/positions'][endIdx:]
                return frames, gt_positions
            else:
                # Load ground truth
                if endIdx>0:
                    gt_positions = f['ground_truth/positions'][startIdx:endIdx:incr]
                else:
                    gt_positions = f['ground_truth/positions'][endIdx:]
                return 0, gt_positions
            
            # # Load frame times
            # start_indices = f['frame_times/start_indices'][:]
            # end_indices = f['frame_times/end_indices'][:]
            # frame_times = list(zip(start_indices, end_indices))
            
            # Get metadata if needed
        #     num_frames = f.attrs['num_frames']
        #     frame_height = f.attrs['frame_height']
        #     frame_width = f.attrs['frame_width']
        # h5_file = h5py.File(filepath, 'r')
        
    
    @staticmethod
    def binary_search_chunks(timestamps, target_time,  side='left', CHUNK_SIZE = 10_000_000):
        low = 0
        high = len(timestamps)
        
        while low < high:
            mid = (low + high) // 2
            
            # Read a small chunk around the midpoint
            chunk_start = max(0, mid - CHUNK_SIZE//2)
            chunk_end = min(len(timestamps), mid + CHUNK_SIZE//2)
            chunk = timestamps[chunk_start:chunk_end]
            
            if side == 'left':
                if chunk[-1] < target_time:
                    low = chunk_end
                elif chunk[0] > target_time:
                    high = chunk_start
                else:
                    # Target is in this chunk
                    idx = np.searchsorted(chunk, target_time, side='left')
                    return chunk_start + idx
            else:  # side == 'right'
                if chunk[-1] < target_time:
                    low = chunk_end
                elif chunk[0] > target_time:
                    high = chunk_start
                else:
                    # Target is in this chunk
                    idx = np.searchsorted(chunk, target_time, side='right')
                    return chunk_start + idx
        
        return low
    
    @staticmethod
    def save_processed_rgbImgs(processed_path, frames, positions):
        """Save processed images and positions into an HDF5 file."""
        with h5py.File(processed_path, "w") as f:
            f.create_dataset("frames", data=frames)
            f.create_dataset("positions", data=positions)

            # f.attrs.update({
            #     'endIdx': endIdx
            # })

    def load_processed_rgbImgs(processed_path, startIdx=0, endIdx=None):
        """Load pre-processed data from HDF5."""
        with h5py.File(processed_path, "r") as f:
            frames = f["frames"][startIdx:endIdx]
            positions = f["positions"][startIdx:endIdx]

            print('Done loading exisiting data')
            
        return frames, positions

def load_hdf5_chunk(args):
        """Helper function to load a chunk of data from HDF5."""
        processed_path, start, end = args
        with h5py.File(processed_path, "r") as f:
            frames = f["frames"][start:end]
            positions = f["positions"][start:end]
        return frames, positions