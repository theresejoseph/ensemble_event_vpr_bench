import os
import csv
from pathlib import Path
# Settings

seq_lens = [1,10,20,30]
dataset_list = ['Brisbane', 'NSAVP']
patch_or_frame = 'frame'  # Options for patch or frame
patch_rows = 2 if patch_or_frame == 'patch' else 1
patch_cols = 2 if patch_or_frame == 'patch' else 1
gpu_methods = [ 'mixvpr', 'netvlad', 'cosplace', 'megaloc']
reps = ['RGB_camera']# ['e2vid','timeSurface', 'eventCount', 'eventCount_noPolarity']  # 'timeSurface' or 'eventCount'
events_per_bins = [100_000, 200_000, 300_000, 500_000, 1_000_000]
time_res_list = [1.0 ]#0.1, 0.15, 0.2, 0.25, 0.5, 
sequences = ["night","morning", "sunrise", "sunset1", "sunset2", "daytime",
                        'R0_FA0', 'R0_FS0', 'R0_FN0', 'R0_RA0', 'R0_RS0', 'R0_RN0']

os.makedirs("hpc/jobs", exist_ok=True)
job_counter = 0


def check_if_result_exists(csv_path, row_dict, key_fields, verbose=False):   
    """
    Check if a result already exists in the CSV file based on key fields.
    """
    def normalize(val):
        if val is None or str(val).lower() in ['nan', 'none', '']:
            return ''
        try:
            f = float(val)
            return str(int(f)) if f.is_integer() else str(f)
        except:
            return str(val).strip()
    
    if not csv_path.exists():
        print(f"CSV file {csv_path} does not exist.")
        return False

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(normalize(row.get(k)) == normalize(row_dict[k]) for k in key_fields):
                return True
    return False




def write_and_submit_job_batch(job_id, job_entries, gpu_use=0):
    """
    Write a PBS job script that runs a batch of experiments (10 at a time).
    Each entry in job_entries is a dict with all CLI args.
    """
    job_script = f"hpc/jobs/job_{job_id}.sh"
    with open(job_script, "w") as f:
        f.write(f"""#!/bin/bash -l
#PBS -N VPR_{job_id}_{dataset_type} 
#PBS -l walltime=03:00:00
#PBS -l mem=32GB
#PBS -l ncpus=4
#PBS -l ngpus={gpu_use}
#PBS -j oe
#PBS -o hpc/outputs/VPR_{dataset_type}_{job_id}.txt
#PBS -e hpc/outputs/VPR_{dataset_type}_{job_id}_err.txt

cd $PBS_O_WORKDIR
conda activate vpr_eval_py310

""")
        for entry in job_entries:
            f.write("python testing.py \\\n")
            for k, v in entry.items():
                f.write(f"  --{k} {v} \\\n")
            f.write("\n")
    
    print(f"🟢 Submitted job {job_id} with GPU={gpu_use}, batch size={len(job_entries)}")
    os.system(f"qsub {job_script}")



job_counter = 0
gpu_use = 0  # Default to CPU jobs
'''Case 1: sweep over all TIME_RES values count_bin = 0, adaptive_bin = 0'''
job_entries = []
max_batch_size = 1
for seq_len in seq_lens:
    for dataset_type in dataset_list:
        experiment_pairs = [ (3,0),(3, 1), (3, 2),(3, 4), (3, 5)] if dataset_type == 'Brisbane' else [(6, 7),(6, 8), (8, 7), (9,11), (10, 11)]
        for method in gpu_methods:
            for time_res in time_res_list:
                for reconstruct_method in reps:
                    for idR, idQ in experiment_pairs:
                        ref_seq, qry_seq = sequences[idR], sequences[idQ]
                        row_dict = {
                            "ref_seq": ref_seq,
                            "qry_seq": qry_seq,
                            "reconstruction_name": reconstruct_method,
                            "vpr_method": method,
                            "seq_len": seq_len,
                            "bin_type": "timebin",
                            "time_res": time_res,
                            "positive_dist_thresh": 25,
                            "patch_or_frame": patch_or_frame,
                            "seq_match_type": 'modified',
                            "patch_rows": patch_rows,
                            "patch_cols": patch_cols,
                        }

                        key_fields = list(row_dict.keys())
                        csv_path = Path(f'./results/vpr_results_{dataset_type}_fixed_timebins_{time_res}.csv')
                        simMatPath = Path(f"./logs/{dataset_type}/fixed_timebins_{time_res}/{ref_seq}_vs_{qry_seq}_{method}_l2_reconstruct_{reconstruct_method}_{time_res}_{patch_or_frame}_{patch_rows}_{patch_cols}.npy")
                        gpu_use = 0 if simMatPath.exists() else 1

                        if check_if_result_exists(csv_path, row_dict, key_fields):
                            print(f"Skipping {ref_seq} vs {qry_seq} with {method} rowcol {patch_rows},{patch_cols}, time {time_res}- already done.")
                            continue

                        job_args = {
                            "dataset_type": dataset_type,
                            "time_res": time_res,
                            "reconstruct_method_name": reconstruct_method,
                            "adaptive_bin": 0,
                            "count_bin": 0,
                            "events_per_bin": 0,
                            "method": method,
                            "patch_or_frame": patch_or_frame,
                            "max_bins": 0,
                            "seq_len": seq_len,
                            "qry_seq_idx": idQ,
                            "ref_seq_idx": idR,
                        }

                        job_entries.append(job_args)

                        if len(job_entries) == max_batch_size:
                            write_and_submit_job_batch(job_counter, job_entries, gpu_use)
                            job_entries = []
                            job_counter += 1

# Submit final batch if it contains < 10 jobs
if job_entries:
    write_and_submit_job_batch(job_counter, job_entries)
print(f"All {job_counter+1} jobs submitted fixed time binning.")




'''Case 2: count_bin = 1 → sweep over all events_per_bin values'''
# job_entries = []
# max_batch_size = 5
# seq_len =1 
# patch_or_frame = 'frame'  # Options for patch or frame
# if patch_or_frame == 'patch':
#     patch_rows = 4
#     patch_cols = 3
# else:
#     patch_rows = 1
#     patch_cols = 1
# for dataset_type in dataset_list:
#     experiment_pairs = [ (3,0),(3, 1), (3, 2),(3, 4), (3, 5)] if dataset_type == 'Brisbane' else [(6, 7),(6, 8), (8, 7), (9,11), (10, 11)]
#     for method in gpu_methods:
#         for events_per_bin in events_per_bins:
#             for reconstruct_method in reps:
#                 for idR, idQ in experiment_pairs:
#                     ref_seq, qry_seq = sequences[idR], sequences[idQ]
#                     row_dict = {
#                         "ref_seq": ref_seq,
#                         "qry_seq": qry_seq,
#                         "reconstruction_name": reconstruct_method,
#                         "vpr_method": method,
#                         "seq_len": seq_len,
#                         "bin_type": "countbin",
#                         "binning_strategy": "fixed",
#                         "events_per_bin": events_per_bin,
#                         "positive_dist_thresh": 25,
#                         "patch_or_frame": patch_or_frame,
#                         "seq_match_type": 'modified',
#                         "patch_rows": patch_rows,
#                         "patch_cols": patch_cols,
#                     }

#                     key_fields = list(row_dict.keys())
#                     csv_path = Path(f'./results/vpr_results_{dataset_type}_fixed_countbins_{events_per_bin}.csv')
#                     simMatPath = Path(f"./logs/{dataset_type}/fixed_countbins_{events_per_bin}/{ref_seq}_vs_{qry_seq}_{method}_l2_reconstruct_{reconstruct_method}_None_{patch_or_frame}_{patch_rows}_{patch_cols}.npy")
#                     if not simMatPath.exists():
#                         gpu_use = 1

#                     if check_if_result_exists(csv_path, row_dict, key_fields, verbose=True):
#                         print(f"Skipping {ref_seq} vs {qry_seq} with {method} - already done.")
#                         continue

#                     job_args = {
#                         "dataset_type": dataset_type,
#                         "time_res": 0,
#                         "reconstruct_method_name": reconstruct_method,
#                         "adaptive_bin": 0,
#                         "count_bin": 1,
#                         "events_per_bin": events_per_bin,
#                         "method": method,
#                         "patch_or_frame": patch_or_frame,
#                         "seq_len": seq_len,
#                         "qry_seq_idx": idQ,
#                         "ref_seq_idx": idR,
#                     }

#                     job_entries.append(job_args)

#                     if len(job_entries) == max_batch_size:
#                         write_and_submit_job_batch(job_counter, job_entries, gpu_use)
#                         job_entries = []
#                         job_counter += 1
#                         gpu_use = 0  # Reset GPU use for next batch
# # Submit final batch if it contains < 10 jobs
# if job_entries:
#     write_and_submit_job_batch(job_counter, job_entries)




# --- Case 3: Adaptive binning (optional, remove if not needed) ---
# for method in gpu_methods:
#     for max_bins in max_bins_list:
#         write_and_submit_single_job(
#             job_counter,
#             method=method,
#             time_res=0,  
#             reconstruct_method=reconstruct_method,
#             adaptive_bin=1,
#             count_bin=0,
#             events_per_bin=0,
#             max_bins=max_bins,
#         )
#         job_counter += 1