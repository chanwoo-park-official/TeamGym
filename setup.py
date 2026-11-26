"""Bound package discovery to code, avoiding local datasets and experiment trees."""

from setuptools import find_namespace_packages, setup


packages = []
for root in ("rllm", "teamgym"):
    packages.append(root)
    packages.extend(
        f"{root}.{name}"
        for name in find_namespace_packages(
            where=root,
            exclude=(
                "tests", "tests.*", "*.tests", "*.tests.*",
                "data.datasets", "data.datasets.*", "run_debug", "run_debug.*",
                "rl.analysis", "rl.analysis.*",
            ),
        )
    )

setup(packages=packages)
