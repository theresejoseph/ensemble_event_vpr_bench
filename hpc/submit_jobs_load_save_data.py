
import os
from pathlib import Path

# === Setup ===
job_dir = Path("hpc/jobs")
job_dir.mkdir(parents=True, exist_ok=True)

# === Configs ===
dataset_type = "Brisbane"  # or "NSAVP"
recon_methods = ['eventCount'] #'eventCount','eventCount_noPolarity', 'timeSurface', 'e2vid' 
event_counts = [1_000_000] ##  1_000_000, 300_000, 500_000, 100_000, 200_000
time_resolutions = [0.2] #0.05, 0.1, 0.25, 0.5, 1.0
max_bins_list = [20,30,40,50,100]  # Maximum number of bins for adaptive binning
experiment_pairs = [ (0,1),(2,3), (4,5)] if dataset_type == 'Brisbane' else [(6, 7),(8,9), (10,11)]

    
base_cmd = "python3 /mnt/hpccs01/home/n10234764/event_vo_vpr/load_and_save.py"

pbs_template = """#!/bin/bash -l
#PBS -N LAS_{dataset_type}_{job_id}
#PBS -l walltime=06:30:00
#PBS -l mem=120GB
#PBS -l ncpus=12
#PBS -l ngpus={use_gpu}
#PBS -j oe
#PBS -o hpc/outputs/LAS_{dataset_type}_{job_id}.txt
#PBS -e hpc/outputs/LAS_{dataset_type}_{job_id}_err.txt

cd $PBS_O_WORKDIR
conda activate vpr_eval_py310

{run_cmd}
"""

def write_and_submit(job_id, recon, time_res, adaptive_bin, count_bin, events_per_bin, max_bins, idR, idQ):
    use_gpu = 1 if recon == 'e2vid' else 0

    run_cmd = f"{base_cmd} " \
              f"--dataset_type {dataset_type} " \
              f"--reconstruct_method_name {recon} " \
              f"--time_res {time_res} " \
              f"--adaptive_bin {adaptive_bin} " \
              f"--count_bin {count_bin} " \
              f"--events_per_bin {events_per_bin} " \
              f"--max_bins {max_bins} " \
              f"--ref_seq_idx {idR} " \
              f"--qry_seq_idx {idQ} "

    job_script = pbs_template.format(dataset_type=dataset_type, job_id=job_id, use_gpu=use_gpu, run_cmd=run_cmd)
    script_path = job_dir / f"job_{job_id}.pbs"
    script_path.write_text(job_script)

    os.system(f"qsub {script_path}")
    print(f"Submitted job_{dataset_type}_{job_id}: recon={recon}, count_bin={count_bin}, time_res={time_res}, epb={events_per_bin}, adaptive={adaptive_bin}, max_bins={max_bins}")

# === Generate Jobs ===
job_id = 0

# # --- Case 1: Fixed time binning ---
for recon in recon_methods:
    for time_res in time_resolutions:
        for idR, idQ in experiment_pairs:
            write_and_submit(
                job_id=job_id,
                recon=recon,
                time_res=time_res,
                adaptive_bin=0,
                count_bin=0,
                events_per_bin=0,
                max_bins=0,
                idR=idR,
                idQ=idQ,
            )
            job_id += 1

# --- Case 2: Count-based binning ---
# for recon in recon_methods:
#     for epb in event_counts:
#         for idR,idQ in experiment_pairs: # 
#             write_and_submit(
#                 job_id=job_id,
#                 recon=recon,
#                 time_res=0,
#                 adaptive_bin=0,
#                 count_bin=1,
#                 events_per_bin=epb,
#                 max_bins=0,  
#                 idR=idR,
#                 idQ=idQ,
#             )
#             job_id += 1

# # --- Case 3: Adaptive binning (optional, remove if not needed) ---
# for recon in recon_methods:
#     for max_bins in max_bins_list:
#         write_and_submit(
#             job_id=job_id,
#             recon=recon,
#             time_res=0,
#             adaptive_bin=1,
#             count_bin=0,
#             events_per_bin=0,
#             max_bins=max_bins
#         )
#         job_id += 1


