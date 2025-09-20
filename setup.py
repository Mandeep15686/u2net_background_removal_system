
"""
Setup script for U²-Net Background Removal System
Team 1: The Isolationists - Subject & Background Separation Specialists
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="u2net-background-removal",
    version="1.0.0",
    author="Team 1: The Isolationists",
    author_email="team1@isolationists.com",
    description="Pixel-perfect background removal using U²-Net deep learning architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isolationists/u2net-background-removal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0", 
            "black>=22.6.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
        ],
        "tensorrt": [
            "tensorrt>=8.4.0",
            "pycuda>=2021.1",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "u2net-train=src.training.train:main",
            "u2net-api=src.api.deployment:main",
            "u2net-inference=src.inference.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="deep-learning computer-vision background-removal u2net segmentation",
    project_urls={
        "Bug Reports": "https://github.com/isolationists/u2net-background-removal/issues",
        "Source": "https://github.com/isolationists/u2net-background-removal",
        "Documentation": "https://u2net-docs.isolationists.com",
    },
)
