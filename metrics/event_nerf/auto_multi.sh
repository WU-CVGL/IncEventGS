#!/bin/bash

# Ensure that the script receives three arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <root_dir> <est> <gt>"
    exit 1
fi

# Assign arguments to variables
root_dir=$1
est=$2
gt=$3

for exp_path in "$root_dir"/*; do
    # echo "$exp_dir"
    base_dir=$(realpath "$exp_path/img_eval")
    exp_name=$(basename "$exp_path")

    # Convert grayscale images to RGB using grey2rgb.py
    echo "Converting grayscale images in $base_dir/$est to RGB..."
    python grey2rgb.py "$base_dir" "$est"

    # Get absolute paths for the est and gt directories
    abs_est=$(realpath "$base_dir/$est"_3ch)
    abs_gt=$(realpath "$base_dir/$gt")

    # Check if both directories exist
    if [ ! -d "$abs_est" ] || [ ! -d "$abs_gt" ]; then
        echo "Error: One of the directories does not exist."
        exit 1
    fi

    # Run computations and save results to .txt files
    echo "*********** Compute PSNR ***********"
    psnr_path=$(realpath "$base_dir/psnr_$est.txt")
    python ./comppsnr.py "$abs_est" "$abs_gt" | tee "$psnr_path"
    echo

    echo "*********** Compute SSIM ***********"
    ssim_path=$(realpath "$base_dir/ssim_$est.txt")
    python ./compssim.py "$abs_est" "$abs_gt" | tee "$ssim_path"
    echo

    echo "*********** Compute LPIPS ***********"
    lpips_result_path=$(realpath "$base_dir/lpips_$est.txt")
    python ./complpips.py "$abs_est" "$abs_gt" | tee "$lpips_result_path"
    echo
done
echo "finished psnr, ssim and lpips computation"


# 完成所有实验后，调用Python脚本生成LaTeX表格
python compose_table.py "$root_dir"
