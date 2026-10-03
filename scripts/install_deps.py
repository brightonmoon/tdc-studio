import subprocess
import sys


def install(pkgs_str):
    pkgs = pkgs_str.strip().split()
    print(f"Installing {len(pkgs)} packages: {pkgs}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)
    print("Done!")

if __name__ == "__main__":
    print("=== Colab Package Installer ===")
    install("rdkit PyTDC torch-geometric lightgbm wandb scikit-learn")
    print("=== Verification ===")
    import rdkit
    import tdc
    import torch
    import torch_geometric
    print("RDKit version:", rdkit.__version__)
    print("TDC version:", getattr(tdc, "__version__", "installed"))
    print("PyG version:", torch_geometric.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
    print("All installations verified successfully!")
