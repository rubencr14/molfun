from setuptools import setup, find_packages

setup(
    name="molfun",
    version="0.1.0",
    description="Optimized Biological Language Models with Triton Kernels",
    author="Molfun Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "triton>=2.0.0",
        "typer>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "molfun = molfun.cli:app",
        ],
    },
    package_dir={"": "."},
    package_data={},
)
