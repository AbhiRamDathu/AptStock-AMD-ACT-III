
"""Manual AMD ROCm environment diagnostic.

Run directly with:
    python amd_engine/amd_engine/test_amd.py

This file intentionally defines no pytest tests.
"""

import time

import torch


def main():
    print("=" * 60)
    print("AptStock AMD Environment Test")
    print("=" * 60)

    print("PyTorch:", torch.__version__)
    print("ROCm:", torch.version.hip)
    print("GPU available:", torch.cuda.is_available())

    if not torch.cuda.is_available() or not torch.version.hip:
        print("SKIP: AMD ROCm GPU is unavailable in this environment.")
        print("Run this diagnostic on the AMD ROCm GPU instance.")
        return 0

    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(4096, 4096, device="cuda")
    y = torch.randn(4096, 4096, device="cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    z = x @ y

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if not torch.isfinite(z).all().item():
        raise RuntimeError("GPU matrix multiplication returned non-finite values.")

    print("Matrix multiplication time:", elapsed)
    print("AMD GPU test: PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
