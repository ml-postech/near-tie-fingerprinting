# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

from setuptools import find_packages, setup

setup(
    name="robust_fp",
    version="1.0.0",
    description="Fingerprinting open-weight language models by watermark distillation on near-tie tokens",
    license="SEE LICENSE IN LICENSE_CODE",
    license_files=["LICENSE_CODE"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
