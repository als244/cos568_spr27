#!/bin/bash
# Record a1/profile_step.py with Nsight Systems, then summarize the recording.
#
#   a1/nsys.sh NAME [profile_step.py arguments]
#
# Writes reports/a1/NAME.nsys-rep   the recording; open it in the Nsight Systems GUI
#        results/a1/NAME_*.csv      summaries: GPU time and GPU metrics per kind of NVTX range,
#                                   GPU time per kernel, and CPU time per CUDA API call
#
# The flags of `nsys profile`:
#   --trace=cuda,nvtx                  record CUDA calls and kernels, and our NVTX ranges
#   --sample=none --cpuctxsw=none      no CPU sampling; we are studying the GPU
#   --gpu-metrics-devices=all          sample the GPU's hardware counters, 10,000 times a second:
#                                      SM activity, tensor-core activity, memory bandwidth, ...
#   --capture-range=cudaProfilerApi    record only between torch.cuda.profiler.start() and stop()
#   --capture-range-end=stop           ... and let the program finish normally afterwards
set -euo pipefail
name="${1:?usage: a1/nsys.sh NAME [profile_step.py arguments]}"
shift
cd "$(dirname "$0")/.."
mkdir -p reports/a1 results/a1

metrics="--gpu-metrics-devices=all"
if nsys profile --gpu-metrics-devices=help 2>&1 | grep -q "Insufficient privilege\|None of the installed GPUs"; then
    echo "note: this node does not let you read the GPU's performance counters;" \
         "recording without GPU metrics."
    metrics=""
fi

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none $metrics \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="reports/a1/$name" --force-overwrite=true \
    python a1/profile_step.py "$@"

nsys stats --quiet --force-export=true --force-overwrite=true \
    --report nvtx_gpu_proj_trace --report cuda_gpu_kern_sum --report cuda_api_sum \
    --format csv --output "results/a1/$name" "reports/a1/$name.nsys-rep"
python a1/nsys_summary.py "results/a1/${name}_nvtx_gpu_proj_trace.csv" "reports/a1/$name.sqlite"
rm -f "reports/a1/$name.sqlite"
echo "wrote reports/a1/$name.nsys-rep and results/a1/${name}_*.csv"
