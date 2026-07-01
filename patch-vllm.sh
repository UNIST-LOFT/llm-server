#!/bin/bash
set -e

cuda_file_path=/root/anaconda3/lib/python3.13/site-packages/vllm
echo "CUDA file path: $cuda_file_path"
sed -i 's| or cls.is_device_capability_family(120)| #or cls.is_device_capability_family(120)|g' "$cuda_file_path/platforms/cuda.py"