import os
import csv
from pathlib import Path
# Settings

seq_len = 1
patch_or_frame = 'patch'  # 'patch' or 'frame'
dataset_type = 'Brisbane'  # 'Brisbane' or 'DVS128'
gpu_methods = [ 'mixvpr']
reps =  ['e2vid','timeSurface', 'eventCount', 'eventCount_noPolarity']  # 'timeSurface' or 'eventCount'
time_res_list = [1.0 ]#[0.05, 0.1, 0.25, 0.5, 1.0 ] #,] #
sequences=["night","morning", "sunrise", "sunset1", "sunset2", "daytime",
     'R0_FA0', 'R0_FS0', 'R0_FN0', 'R0_RA0', 'R0_RS0', 'R0_RN0', "night_training",
     "morning_training", "sunrise_training", "sunset1_training", "sunset2_training", "daytime_training",]
os.makedirs("hpc/jobs", exist_ok=True)
job_counter = 0





def normalize_value(val):
    if val in [None, '', 'nan']:
        return ''
    try:
        return str(int(float(val))) if float(val).is_integer() else str(float(val))
    except:
        return str(val).strip()

def check_if_result_exists(csv_path, row_dict, key_fields, verbose=False):
    """
    Check if a result already exists in the CSV file based on key fields.
    Normalize values for robust comparison.
    """
    if not csv_path.exists():
        return False

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(
                normalize_value(row.get(k)) == normalize_value(row_dict[k])
                for k in key_fields
            ):
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
#PBS -N PGS_{job_id}_{dataset_type} 
#PBS -l walltime=01:00:00
#PBS -l mem=32GB
#PBS -l ncpus=4
#PBS -l ngpus={gpu_use}
#PBS -j oe
#PBS -o hpc/outputs/PGS_{dataset_type}_{job_id}.txt
#PBS -e hpc/outputs/PGS_{dataset_type}_{job_id}_err.txt

cd $PBS_O_WORKDIR
conda activate vpr_eval_py310

""")
        for entry in job_entries:
            f.write("python testing.py \\\n")
            for k, v in entry.items():
                f.write(f"  --{k} {v} \\\n")
            f.write("\n")
    
    print(f"🟢 Submitted job {job_id} with GPU={gpu_use}, batch size={len(job_entries)}")
    # os.system(f"qsub {job_script}")



job_counter = 0
gpu_use = 0  # Default to CPU jobs
'''Case 1: count_bin = 0 → sweep over all time_res values'''
job_entries = []
max_batch_size = 10

experiment_pairs = [ (15,12),(15, 13), (15, 14), (15, 16), (15, 17)]
for method in gpu_methods:
    for time_res in time_res_list:
        for reconstruct_method in reps:
            for idR, idQ in experiment_pairs:
                for patch_rows in range(1, 10):
                    for patch_cols in range(1, 10):

                        ref_seq, qry_seq = sequences[idR], sequences[idQ]
                        row_dict = {
                            "ref_seq": ref_seq,
                            "qry_seq": qry_seq,
                            "reconstruction_name": reconstruct_method,
                            "vpr_method": method,
                            "seq_len": seq_len,
                            "bin_type": "timebin",
                            "binning_strategy": "fixed",
                            "events_per_bin": '',
                            "time_res": time_res,
                            "positive_dist_thresh": 25,
                            "patch_or_frame": patch_or_frame,
                            "seq_match_type": 'modified',
                            "patch_rows": patch_rows,
                            "patch_cols": patch_cols,
                        }

                        key_fields = list(row_dict.keys())
                        csv_path = Path(f'./hpc/patch_grid_search_results1.csv')

                        if check_if_result_exists(csv_path, row_dict, key_fields):
                            print(f"Skipping {ref_seq} vs {qry_seq} with {method} - already done.")
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
                            "patch_num_rows": patch_rows,
                            "patch_num_cols": patch_cols,
                        }

                        job_entries.append(job_args)

                        if len(job_entries) == max_batch_size:
                            write_and_submit_job_batch(job_counter, job_entries, gpu_use)
                            job_entries = []
                            job_counter += 1

                            # assert False

# Submit final batch if it contains < 10 jobs
if job_entries:
    write_and_submit_job_batch(job_counter, job_entries)

