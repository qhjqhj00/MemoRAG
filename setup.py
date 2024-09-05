from setuptools import setup, find_packages

setup(
    name='memorag',
    version='0.1',
    description='A Python package for memory-augmented retrieval-augmented generation',
    author='Tommy Chien',
    author_email='tommy@chien.io',
    packages=find_packages(),
    install_requires=[],  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)