"""Check that PyTorch sees the GPU and can run a computation on it."""

import torch


def main() -> None:
    print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA not available: training would run on CPU.")
        return
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name} (sm_{capability[0]}{capability[1]}, {vram_gb:.1f} GB)")

    x = torch.randn(1024, 1024, device="cuda")
    print(f"matmul on GPU ok: {(x @ x).sum().item():.2f}")


if __name__ == "__main__":
    main()
