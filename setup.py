#!/usr/bin/env python3
"""
Setup script for Ventamin AI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="ventamin-ai",
    version="1.0.0",
    author="Ventamin AI Team",
    author_email="contact@ventamin.ai",
    description="Intelligent video generation system with AI-powered analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ventamin-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.12.0",
        "moviepy>=2.1.0",
        "Pillow>=11.0.0",
        "numpy>=2.1.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "ai": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "openai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ventamin-ai=start_ventamin_ai:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.ini", "*.json"],
    },
)
