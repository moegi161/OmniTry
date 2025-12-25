#!/bin/bash

base="/mnt/workspace2024/chan/OmniTry/results/celebv-hq-100"

# loop over subfolders
for dir in "$base"/*/; do
  # strip trailing slash
  folder_name=$(basename "$dir")
  mp4_path="$base/${folder_name}.mp4"

  # check if mp4 already exists
  if [ ! -f "$mp4_path" ]; then
    echo "Missing video for: $folder_name"
    frames_out_dir="$dir"
    video_out="$mp4_path"

    # make the video
    ffmpeg -y -r 30 -i "${frames_out_dir}/frame_%4d.png" \
      -vcodec libx264 -crf 11 -pix_fmt yuv420p "$video_out"
  else
    echo "Already has mp4: $folder_name"
  fi
done
