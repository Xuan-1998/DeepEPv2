from ._build_info import EXPECTED_CUDA_MAJOR as _EXPECTED_CUDA_MAJOR
from .prerequisites import check_prerequisites as _check_prerequisites

if _EXPECTED_CUDA_MAJOR is not None:
    _check_prerequisites(expected_cuda_major=_EXPECTED_CUDA_MAJOR)
del _EXPECTED_CUDA_MAJOR, _check_prerequisites
