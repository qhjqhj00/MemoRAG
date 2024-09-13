from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='memorag',
    version='0.1.4',
    description='A Python package for memory-augmented retrieval-augmented generation',
    author='Tommy Chien',
    author_email='tommy@chien.io',
    packages=find_packages(),
    install_requires=requirements,  
    classifiers=[
        'Programming Language :: Python :: 3.10',  
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  
)
