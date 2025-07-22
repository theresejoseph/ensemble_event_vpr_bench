#!/bin/bash

mkdir -p gifs_cropped

filelocations=(
"/mnt/hpccs01/home/n10234764/data/BrisbaneEvent/image_reconstructions/fixed_timebins_0.2/e2vid/daytime_e2vid_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/BrisbaneEvent/image_reconstructions/fixed_timebins_0.2/eventCount_noPolarity/daytime_eventCount_noPolarity_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/BrisbaneEvent/image_reconstructions/fixed_timebins_0.2/eventCount/daytime_eventCount_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/BrisbaneEvent/image_reconstructions/fixed_timebins_0.2/timeSurface/daytime_timeSurface_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/NSAVP/image_reconstructions/fixed_timebins_0.2/e2vid/R0_FA0_e2vid_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/NSAVP/image_reconstructions/fixed_timebins_0.2/eventCount_noPolarity/R0_FA0_eventCount_noPolarity_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/NSAVP/image_reconstructions/fixed_timebins_0.2/eventCount/R0_FA0_eventCount_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/NSAVP/image_reconstructions/fixed_timebins_0.2/timeSurface/R0_FA0_timeSurface_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/NSAVP/image_reconstructions/fixed_timebins_0.2/e2vid/R0_RN0_e2vid_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/NSAVP/image_reconstructions/fixed_timebins_0.2/eventCount_noPolarity/R0_RN0_eventCount_noPolarity_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/NSAVP/image_reconstructions/fixed_timebins_0.2/eventCount/R0_RN0_eventCount_0.2.mp4"
"/mnt/hpccs01/home/n10234764/data/NSAVP/image_reconstructions/fixed_timebins_0.2/timeSurface/R0_RN0_timeSurface_0.2.mp4"
)

for file in "${filelocations[@]}"; do
  duration=$(ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0 "$file")
  start=$(echo "$duration * 0.5" | bc -l)
  length=$(echo "$duration * 0.05" | bc -l)

  start_fmt=$(printf "%.2f" "$start")
  length_fmt=$(printf "%.2f" "$length")

  base=$(basename "${file%.mp4}")
  gif_out="./plots/gifs_cropped/${base}_last10.gif"

  ffmpeg -hide_banner -loglevel error -ss "$start_fmt" -t "$length_fmt" -i "$file" \
         -vf "fps=5,scale=480:-1:flags=lanczos" "$gif_out"

  echo "Saved $gif_out"
done
