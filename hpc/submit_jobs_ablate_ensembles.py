import os
import csv
from pathlib import Path
from itertools import product
import pandas as pd

def load_existing_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: Could not load existing CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def normalize_field(val):
    """Standardize a CSV field: strip, sort comma-lists, lowercase"""
    if pd.isna(val) or val == '':
        return ''
    val = str(val).strip().lower()
    if ',' in val:
        return ','.join(sorted(part.strip() for part in val.split(',')))
    return val

def is_duplicate(row: dict, existing_df: pd.DataFrame) -> bool:
    if existing_df.empty:
        return False

    for _, existing_row in existing_df.iterrows():
        match = True
        for key in ['ref_name', 'qry_name', 'ensemble_over', 'vpr_methods', 'recon_methods', 'time_strs']:
            if key not in existing_row or normalize_field(existing_row[key]) != normalize_field(row[key]):
                match = False
                break
        if match:
            return True
    return False


os.makedirs("hpc/jobs", exist_ok=True)
job_counter = 0
max_batch_size = 5
gpu_use = 0
ref_qry_pairs = [
    ('R0_FA0', 'R0_FS0'),
    ('R0_FA0', 'R0_FN0'),
    ('R0_FN0', 'R0_FS0'),
    ('R0_RA0', 'R0_RN0'),
    ('R0_RS0', 'R0_RN0'),
]
dataset = "NSAVP"
# dataset = "Brisbane" 
# ref_qry_pairs = [
#     ('sunset1', 'night'),
#     ('sunset1', 'morning'),
#     ('sunset1', 'sunrise'),
#     ('sunset1', 'sunset2'),
#     ('sunset1', 'daytime'),
# ]
slens = [10, 20, 30]  # Fixed sequence lengths
vpr_all = ['mixvpr', 'megaloc', 'cosplace', 'netvlad']
recon_all = ['RGB_camera','e2vid', 'eventCount', 'eventCount_noPolarity', 'timeSurface']
time_all = [0.1,0.25,0.5,1.0]
ensemble_groups = ['recon', 'vpr', 'patch']  # 'recon', 'vpr', 'time', 'patch'
csv_path = Path(f'./hpc/ablate_ensemble_combination_noSubsamp.csv')
existing_df = load_existing_results(csv_path)


def write_and_submit_job_batch(job_id, job_entries):
    job_script = f"hpc/jobs/job_ensemble_{job_id}.sh"
    with open(job_script, "w") as f:
        f.write(f"""#!/bin/bash -l
#PBS -N ENS_{job_id}_{dataset}
#PBS -l walltime=02:00:00
#PBS -l mem=48GB
#PBS -l ncpus=12
#PBS -l ngpus={gpu_use}
#PBS -j oe
#PBS -o hpc/outputs/ENS_{dataset}_{job_id}.txt
#PBS -e hpc/outputs/ENS_{dataset}_{job_id}_err.txt

cd $PBS_O_WORKDIR
conda activate vpr_eval_py310

""")
        for entry in job_entries:
            f.write("python ablate_ensembles.py \\\n")
            for k, v in entry.items():
                f.write(f"  --{k} '{v}' \\\n")
            f.write("\n")

    print(f"🟢 Wrote job {job_id} with {len(job_entries)} entries")
    os.system(f"qsub {job_script}")

# Generate jobs
job_entries = []
for seq_len in slens:  # Fixed sequence length
    for ref, qry in ref_qry_pairs:
        for group in ensemble_groups:
            if group == "vpr":
                vprs = [",".join(vpr_all)]
                recons = recon_all
                times = time_all
            elif group == "recon":
                vprs = vpr_all
                recons = [",".join(recon_all)]
                times = time_all
            elif group == "time":
                vprs = vpr_all
                recons = recon_all
                times = [",".join(map(str, time_all))]
            else:  # patch: no ensemble, individual only
                vprs = vpr_all
                recons = recon_all
                times = time_all

            for v, r, t in product(vprs, recons, times):
                result_row = {
                    'ref_name': ref,
                    'qry_name': qry,
                    'ensemble_over': group,
                    'vpr_methods': v,
                    'recon_methods': r,
                    'time_strs': str(t),
                    'seqLen': seq_len,
                }

                if is_duplicate(result_row, existing_df):
                    print(f"⏭️ Skipping existing job: {result_row}")
                    continue
                entry = {
                    "dataset": dataset,
                    "ref_seq": ref,
                    "qry_seq": qry,
                    "ensemble_over": group,
                    "vpr_methods": v,
                    "recon_methods": r,
                    "time_strs": str(t),
                    "seq_len": seq_len }
                job_entries.append(entry)

                if len(job_entries) == max_batch_size:
                    write_and_submit_job_batch(job_counter, job_entries)
                    job_counter += 1
                    job_entries = []
                    # assert False 

if job_entries:
    write_and_submit_job_batch(job_counter, job_entries)
