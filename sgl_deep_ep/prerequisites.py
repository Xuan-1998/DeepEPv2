import platform

_README_URL = ("https://github.com/sgl-project/DeepEP/tree/sgl-deepep/sgl_deep_ep")


def _failure(message: str) -> ImportError:
    return ImportError(f"{message}. See {_README_URL} for host prerequisites")


def check_prerequisites(*, expected_cuda_major: int) -> None:
    if platform.system() != "Linux":
        raise _failure("sgl-deep-ep supports Linux only")

    machine = platform.machine()
    if machine not in {"x86_64", "aarch64"}:
        raise _failure(f"sgl-deep-ep does not support architecture {machine}")

    try:
        import torch
    except ImportError as error:
        raise _failure("PyTorch is not installed") from error

    torch_cuda = torch.version.cuda
    if not torch_cuda:
        raise _failure("The installed PyTorch build does not provide CUDA")

    try:
        torch_cuda_major = int(torch_cuda.split(".", 1)[0])
    except (TypeError, ValueError) as error:
        raise _failure(f"Cannot parse PyTorch CUDA version {torch_cuda!r}") from error

    if torch_cuda_major != expected_cuda_major:
        raise _failure(f"This wheel requires CUDA {expected_cuda_major}, but PyTorch uses CUDA {torch_cuda_major}")

    if not torch.cuda.is_available():
        raise _failure("The NVIDIA driver does not expose a usable CUDA device")
