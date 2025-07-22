#!/bin/bash -l

#PBS -N ablate_ensembles
#PBS -l walltime=3:30:00
#PBS -l mem=100GB
#PBS -l ncpus=8
#PBS -l ngpus=0
#PBS -o hpc/outputs/ablate_ensembles_out.txt
#PBS -e hpc/outputs/ablate_esemb_err.txt

cd $PBS_O_WORKDIR
conda activate vpr_eval_py310

python3 /mnt/hpccs01/home/n10234764/event_vo_vpr/ablate_ensembles.py
# Define sequence indices to iterate over
# ref_ids=(3)
# qry_ids=(0 1 2 4 5)

# # Loop over all combinations
# for ref in "${ref_ids[@]}"; do
#   for qry in "${qry_ids[@]}"; do
#     echo "Running ref=$ref qry=$qry"
#     python3 /mnt/hpccs01/home/n10234764/event_vo_vpr/testing.py \
#       --reconstruct_method_name RGB_camera \
#       --dataset_type NSAVP \
#       --ref_seq_idx "$ref" \
#       --qry_seq_idx "$qry" \
#       --adaptive_bin 0 \
#       --time_res 1.0 \
#       --patch_or_frame frame 
#   done
# done   

# python3 /mnt/hpccs01/home/n10234764/event_vo_vpr/testing.py --adaptive_bin 0 --time_res 1.0 --reconstruct_method_name RGB_camera --dataset_type Brisbane --ref_seq_idx 2 --qry_seq_idx 1 
# python3 /mnt/hpccs01/home/n10234764/event_vpr/load_and_save.py --dataset_type NSAVP --reconstruct_method_name EventTwoChannelCount --time_res 0.1 --random_downsample_bool 0

# python3 /mnt/hpccs01/home/n10234764/event_vpr/load_and_save.py --dataset_type Brisbane --reconstruct_method_name TwoChannelEventFrequency --time_res 1.0
# python3 /mnt/hpccs01/home/n10234764/event_vpr/load_and_save.py --dataset_type NSAVP --reconstruct_method_name RGB_camera

# python3 /mnt/hpccs01/home/n10234764/event_vpr/load_and_save.py --dataset_type Brisbane --reconstruct_method_name TwoChannelEventFrequency --time_res 1.0 --end_idx 100

# Run your command (modify this as needed)
# python3 /mnt/hpccs01/home/n10234764/event_vpr/run_place_recV2.py --dataset_type Brisbane --reconstruct_method_name EventUnsignedCount --rerun_place_rec 1
# python3 /mnt/hpccs01/home/n10234764/event_vpr/run_place_recV2.py --dataset_type NSAVP --reconstruct_method_name EventFrequency --rerun_place_rec 1 

# traverse_to_name = {
#     'dvs_vpr_2020-04-21-17-03-03': 'sunset1',
#     'dvs_vpr_2020-04-22-17-24-21': 'sunset2',
#     'dvs_vpr_2020-04-24-15-12-03': 'daytime',
#     'dvs_vpr_2020-04-28-09-14-11': 'morning',
#     'dvs_vpr_2020-04-29-06-20-23': 'sunrise',
#     'dvs_vpr_2020-04-27-18-13-29': 'night'
# }
# https://zenodo.org/records/4302805/files/20200427_181204-night_concat.nmea?download=1
# wget https://huggingface.co/datasets/TobiasRobotics/brisbane-event-vpr/resolve/main/dvs_vpr_2020-04-27-18-13-29.nmea -O night.nmea
# wget "https://zenodo.org/records/4302805/files/20200427_181204-night_concat.nmea?download=1" -O night.nmea