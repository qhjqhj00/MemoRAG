from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="memorag",
    version="0.1",
    description="A Python package for memory-augmented retrieval-augmented generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tommy Chien",
    author_email="tommy@chien.io",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
