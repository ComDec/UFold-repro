#!/usr/bin/env python
"""Verify that the runtime environment has all required packages
and that the UFold model can perform a forward pass."""

import sys

def check_imports():
    """Import every package the project needs. Returns (torch_module, ok)."""
    print(f"Python {sys.version}")
    missing = []
    torch_mod = None

    checks = [
        ("numpy",       "np",    lambda m: m.__version__),
        ("scipy",       None,    lambda m: m.__version__),
        ("sklearn",     None,    lambda m: m.__version__),
        ("torch",       None,    lambda m: m.__version__),
        ("munch",       None,    lambda m: m.__version__),
        ("yaml",        None,    lambda m: m.__version__),
        ("cv2",         None,    lambda m: m.__version__),
        ("torcheval",   None,    lambda m: m.__version__),
        ("matplotlib",  None,    lambda m: m.__version__),
    ]

    display_names = {
        "np": "numpy", "sklearn": "scikit-learn",
        "yaml": "pyyaml", "cv2": "opencv",
    }

    for mod_name, alias, version_fn in checks:
        label = display_names.get(mod_name, mod_name)
        try:
            mod = __import__(mod_name)
            ver = version_fn(mod)
            print(f"  {label:14s} {ver}")
            if mod_name == "torch":
                torch_mod = mod
        except ImportError:
            missing.append(label)
            print(f"  {label:14s} MISSING")

    if missing:
        print(f"\nWARNING: missing packages: {', '.join(missing)}")
        print("Install them before running the project.")
        return None, False

    return torch_mod, True


def check_cuda(torch):
    """Report CUDA availability."""
    if torch.cuda.is_available():
        print(f"\nCUDA available: {torch.version.cuda}  "
              f"(device: {torch.cuda.get_device_name(0)})")
    else:
        print("\nCUDA not available -- running on CPU only")


def check_model(torch):
    """Instantiate U_Net and run a small forward pass."""
    from Network import U_Net

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = U_Net(img_ch=17, output_ch=1).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nU_Net created ({param_count:,} parameters) on {device}")

    # The model uses MaxPool2d with kernel_size=2 five times,
    # so the spatial dimension must be divisible by 2^5 = 32.
    L = 64
    x = torch.randn(1, 17, L, L, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"  Forward pass: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (1, L, L), f"Unexpected output shape: {y.shape}"


def main():
    print("=" * 50)
    print("UFold environment check")
    print("=" * 50)

    torch, imports_ok = check_imports()

    if torch is not None:
        check_cuda(torch)
        check_model(torch)

    print()
    print("=" * 50)
    if imports_ok:
        print("Environment OK")
    else:
        print("Environment INCOMPLETE -- see warnings above")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()
