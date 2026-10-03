import rdkit
import tdc
import torch

print("=== Colab Environment Verification ===")
print("TDC Version:", getattr(tdc, "__version__", "installed"))
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
print("RDKit Version:", rdkit.__version__)
print("PyTorch Version:", torch.__version__)
print("All check passed!")
