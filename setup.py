from setuptools import setup, find_packages

setup(
    name="clv_causal",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
