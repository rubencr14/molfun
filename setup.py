from setuptools import setup, find_packages

setup(
    name="molfun",
    version="0.2.0",
    description="Fine-tuning, modular architecture and GPU acceleration for molecular ML models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Rubén Cañadas",
    url="https://github.com/rubencr14/molfun",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "typer>=0.9.0",
        "biopython>=1.80",
        "numpy>=1.24.0",
    ],
    extras_require={
        "kernels": [
            "triton>=2.0.0",
        ],
        "openfold": [
            "dm-tree",
            "ml-collections",
        ],
        "peft": [
            "peft>=0.6.0",
        ],
        "agents": [
            "openai>=1.0.0",
        ],
        "agents-anthropic": [
            "anthropic>=0.20.0",
        ],
        "agents-ollama": [
            "ollama>=0.1.0",
        ],
        "agents-litellm": [
            "litellm>=1.0.0",
        ],
        "streaming": [
            "fsspec>=2023.1.0",
            "s3fs>=2023.1.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "comet": [
            "comet-ml>=3.30.0",
        ],
        "mlflow": [
            "mlflow>=2.0.0",
        ],
        "langfuse": [
            "langfuse>=2.0.0",
        ],
        "hub": [
            "huggingface_hub>=0.19.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "."},
    package_data={},
)
