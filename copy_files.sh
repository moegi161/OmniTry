# paths
src="/mnt/workspace2024/chan/OmniTry/results/celebv-hq"
dst="/mnt/workspace2024/chan/OmniTry/results/celebv-hq-100"

mkdir -p "$dst"

# PREVIEW: list which folders would be copied (no changes yet)
find "$src" -mindepth 1 -maxdepth 1 -type d -print0 |
while IFS= read -r -d '' d; do
  n=$(find "$d" -maxdepth 1 -type f | wc -l)
  if [ "$n" -eq 120 ]; then
    echo "Will copy: $d  (files: $n)"
  fi
done

# If the preview looks right, run the actual copy:
find "$src" -mindepth 1 -maxdepth 1 -type d -print0 |
while IFS= read -r -d '' d; do
  n=$(find "$d" -maxdepth 1 -type f | wc -l)
  if [ "$n" -eq 120 ]; then
    rsync --progress -ah "$d" "$dst"/
  fi
done
