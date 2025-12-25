#!/bin/bash
# Run inference.py for multiple (person, reference) pairs

# Root directory of reference images
REF_ROOT="/mnt/workspace2024/chan/celebv-hq/reference_100"

# Output base directory
OUT_ROOT="results"

# Pair 1: Eyeglasses
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --person /mnt/workspace2024/chan/celebv-hq/random_picks/random_120/frames_align/9TGhGbQeLKI_22_0/frame_0060.png \
  --ref $REF_ROOT/ref_img/eyeglasses_316.png \
  --class glasses \
  --frames-out $OUT_ROOT/9TGhGbQeLKI_22_0_out \
  --steps 20 --guidance 30 --seed 1234 

# Pair 2: Sunglasses
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --person /mnt/workspace2024/chan/celebv-hq/random_picks/random_120/frames_align/eQONGI5N0dM_4/frame_0021.png \
  --ref $REF_ROOT/ref_img/sunglasses_000.png \
  --class sunglasses \
  --frames-out $OUT_ROOT/eQONGI5N0dM_4_out \
  --steps 20 --guidance 30 --seed 1234 

# Pair 3: Eyeglasses
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --person /mnt/workspace2024/chan/diffae/editing_exemplar/1113_swap_remove_test/0U3lKMXzNp8_7/frame_0012.png \
  --ref $REF_ROOT/ref_img/eyeglasses_061.png \
  --class glasses \
  --frames-out $OUT_ROOT/0U3lKMXzNp8_7_out \
  --steps 20 --guidance 30 --seed 1234 

# Pair 4: Sunglasses
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --person /mnt/workspace2024/chan/diffae/editing_exemplar/0522_swap/frames_align/-YKboJG0xdo_68_0/frame_0086.png \
  --ref $REF_ROOT/ref_img/sunglasses_049.png \
  --class sunglasses \
  --frames-out $OUT_ROOT/-YKboJG0xdo_68_0_out \
  --steps 20 --guidance 30 --seed 1234 
