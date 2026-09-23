"""Remote script to check CUDA GPU on Google Colab."""

import sys
import torch

print("=== Colab Cloud VM Diagnostic ===")
print("Python:", sys.version)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
    print("VRAM Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
