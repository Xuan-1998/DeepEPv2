# sgl-deep-ep

`sgl-deep-ep` is SGLang's binary distribution of
[DeepEP](https://github.com/sgl-project/DeepEP). The distribution name is
`sgl-deep-ep`, while the Python import remains compatible with DeepEP:

```python
import deep_ep
```

## Installation

The PyPI package targets CUDA 13 and installs with:

```bash
pip install sgl-deep-ep
```

Each wheel is specific to a CPython version, CPU architecture, and CUDA major.
The package checks these constraints when `deep_ep` is imported.

## Host prerequisites

The wheel contains DeepEP's Python code and CUDA extension. It does not and
cannot provision the host kernel or RDMA stack. Every host must provide:

- a compatible NVIDIA driver and CUDA runtime;
- the RDMA/NVLink environment required by the selected DeepEP transport.

At import time, the package checks the supported platform, PyTorch and CUDA
versions, and CUDA driver/device availability. Failures raise an actionable
`ImportError`. Transport-specific prerequisites are validated when DeepEP
initializes the selected transport.

CUDA 13 wheels use DeepEP v2 and its NCCL Gin transport. Release and runtime
images pin `nvidia-nccl-cu13==2.30.7`. NVSHMEM remains a build and runtime
dependency because the current v2 extension still links the legacy DeepEP methods.

## Wheel release workflow

Wheels are built and published by SGLang's
[Release sgl-deep-ep workflow](https://github.com/sgl-project/sglang/actions/workflows/release-whl-deepep.yml).
Implementation and packaging are maintained together on `sgl-deepep`.
Before starting a release, merge both kinds of changes into that branch:

- CUDA 13, x86_64 and aarch64: `sgl-deepep` (DeepEP v2)

Run the workflow manually with:

- `version`: the public package version, without a leading `v`;
- `deepep-ref`: the branch, tag, or commit containing both the implementation
  and `sgl_deep_ep/` (normally `sgl-deepep`; use a commit for a reproducible release).


For each selected CUDA target, the workflow uploads CUDA-tagged wheels to the `cu130` index in
[sgl-project/whl](https://github.com/sgl-project/whl). CUDA 13.0 wheels are also
published to PyPI as `sgl-deep-ep`, without a CUDA suffix in the package name.
Publishing requires the `GH_PAT_FOR_WHL_RELEASE` and
`SGL_DEEP_EP_PYPI_TOKEN` repository secrets.

After the workflow completes, check that every expected Python and architecture
artifact was published, install the appropriate wheel in a clean environment,
and verify both `import deep_ep` and a multi-GPU DeepEP or SGLang workload.

## Building from a checkout

Initialize the submodules and install the build dependencies, including the
target CUDA PyTorch build, NCCL 2.30.7, NVSHMEM, ninja, setuptools, and wheel.
The release container in SGLang's `docker/sgl-deep-ep.Dockerfile` provides these
dependencies for manylinux builds. Then run:

```bash
git clone --recursive --branch sgl-deepep https://github.com/sgl-project/DeepEP.git
cd DeepEP
SGL_DEEP_EP_VERSION=0.1.3 bash sgl_deep_ep/build_sgl_deep_ep.sh dist 13.0 x86_64
```

The arguments are the output directory, CUDA version (`13.0` or `13.4`), and
CPU architecture (`x86_64` or `aarch64`). Use the CUDA version and
architecture of the build environment. The script reads implementation and
packaging from its own checkout, stages a temporary copy, and leaves the source
unchanged. `SGL_DEEP_EP_VERSION` overrides `sgl_deep_ep/VERSION`; the wheel
version includes the corresponding `+cu130` or `+cu134` suffix.
`PYTHON_BIN` selects the build interpreter and `MAX_JOBS` controls parallelism.

This produces a native Linux wheel. The SGLang release build script applies
`auditwheel repair` for manylinux and creates the public-version PyPI wheel.
The CUDA 13.4 image also uses this entry point; the release workflow currently
publishes CUDA 13.0 wheels only.
