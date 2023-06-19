import subprocess
import torch

if not torch.cuda.is_available():
    print("Ooops! No CUDA")

else:
    CUDA_version = [
        s
        for s in subprocess.check_output(["nvcc", "--version"])
        .decode("UTF-8")
        .split(", ")
        if s.startswith("release")
    ][0].split(" ")[-1]
    print("Cuda OK")
    print("CUDA version:", CUDA_version)
