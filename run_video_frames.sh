# Get the current working directory
current_dir=$(pwd)

attribute="eyeglasses"

# Specify the relative paths
object_ref_path_relative="/mnt/workspace2024/chan/celebv-hq/reference_100/ref_img/*.png"
object_mask_path_relative="/mnt/workspace2024/chan/celebv-hq/reference_100/ref_mask" # This is a directory
target_path_relative="/mnt/workspace2024/chan/celebv-hq/celebv-hq-100/frames" # This is a directory
target_mask_path_relative="/mnt/workspace2024/chan/celebv-hq/reference_100/ref_target_mask" # This is a directory
save_dir_relative="results/celebv-hq-100"

# Combine the current directory with the relative paths
object_ref_path="$object_ref_path_relative"
object_mask_path="$object_mask_path_relative"
target_path="$target_path_relative"
target_mask_path="$target_mask_path_relative"
save_dir="$current_dir/$save_dir_relative"
mkdir -p $save_dir

# Path to the text file containing video_id and ref_id
input_file="/mnt/workspace2024/chan/celebv-hq/celebv-hq-100/dataset_pairs.txt"

# Read the input file line by line
while IFS=', ' read -r video_path ref_path; do
    # Extract video_id and ref_id from the paths
    video_id=$(basename "$video_path")
    ref_id=$(basename "$ref_path" .png)
    echo "video_id: $video_id, ref_id: $ref_id"
    
    # Construct the paths
    object_ref_path="/mnt/workspace2024/chan/celebv-hq/reference_100/ref_img/${ref_id}.png"
    new_object_ref_path="$object_ref_path"
    new_object_mask_path="$object_mask_path/${ref_id}.png"
    new_target_mask_path="$target_mask_path/${ref_id}.png"
    new_target_path="$target_path/$video_id"
    
    # Extract base names for dynamic file naming
    object_basename=$(basename "$object_ref_path" .png)
    target_basename=$video_id
    
    # Function to generate the desired filename
    generate_filename() {
        local index=$1
        echo "${object_basename}_${target_basename}"
    }
    
    # Define the filenames
    filename=$(generate_filename omnitry)

    
    # Check for Python path through Poetry or directly
    timestamp=$(date +%s)

    if [[ $ref_id == *"eyeglasses_04"* ]]; then
        echo "object_ref_path: $new_object_ref_path"
        echo "object_mask_path: $new_object_mask_path"
        echo "target_mask_path: $new_target_mask_path"

        echo "object_base_name: $object_basename"
        echo "target_base_name: $target_basename"
        echo "target_path: $new_target_path"
            start_time=$SECONDS
            
        CUDA_VISIBLE_DEVICES=0 python inference.py \
        --person "$new_target_path" \
        --ref "$new_object_ref_path" \
        --mask "$new_object_mask_path" \
        --class glasses \
        --frames-out $save_dir/$filename \
        --video-out $save_dir/$filename.mp4 \
        --steps 20 --guidance 30 --seed 1234 
        duration=$(( SECONDS - start_time ))
        echo "Time for model5_mimic-brush.py: ${duration} seconds"
    fi

done < "$input_file"