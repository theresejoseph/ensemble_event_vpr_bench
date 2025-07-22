#!/bin/bash

mkdir -p gifs_cropped

for file in *.mp4; do
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
  half_duration=$(echo "$duration / 2" | bc -l)
  half_duration_fmt=$(printf "%.2f" "$half_duration")
  
  base="${file%.mp4}"
  ffmpeg -ss "$half_duration_fmt" -i "$file" -t "$half_duration_fmt" -vf "fps=10,scale=320:-1:flags=lanczos" "gifs_cropped/${base}_second_half.gif"
done
