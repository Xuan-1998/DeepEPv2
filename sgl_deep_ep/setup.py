import importlib.metadata
import os
import re
import runpy
from pathlib import Path

import setuptools

_OVERRIDDEN_REQUIREMENT = re.compile(
    r"\s*(?:torch|pynvml|nvidia[-_.]ml[-_.]py)(?:\s|[<>=!~;@\[]|$)",
    re.IGNORECASE,
)


def _installed_torch_requirement() -> str:
    try:
        installed_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("PyTorch must be installed before building sgl-deep-ep") from exc

    public_version = installed_version.partition("+")[0]
    return f"torch=={public_version}"


def _run_deepep_setup() -> None:
    version = os.environ.get("SGL_DEEP_EP_VERSION")
    if not version:
        raise RuntimeError("SGL_DEEP_EP_VERSION must be set")

    original_setup = setuptools.setup

    def sgl_setup(**kwargs):
        requirements = [
            requirement for requirement in kwargs.get("install_requires") or [] if not _OVERRIDDEN_REQUIREMENT.match(requirement)
        ]
        requirements.extend(["nvidia-ml-py", _installed_torch_requirement()])

        kwargs.update(
            name="sgl-deep-ep",
            version=version,
            python_requires=">=3.10",
            long_description=Path("README.md").read_text(),
            long_description_content_type="text/markdown",
            install_requires=requirements,
            license_files=("LICENSE", ),
        )
        return original_setup(**kwargs)

    setuptools.setup = sgl_setup
    try:
        runpy.run_path("_deepep_setup.py", run_name="__main__")
    finally:
        setuptools.setup = original_setup


if __name__ == "__main__":
    _run_deepep_setup()
