""" SeismiPro is a library for seismic data processing. """

from setuptools import setup, find_packages
import re

with open('seismicpro/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='SeismicPro',
    packages=find_packages(exclude=['tutorials', 'docker_containers', 'datasets', 'models']),
    version=version,
    url='https://github.com/analysiscenter/SeismicPro',
    license='Apache License 2.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='A framework for seismic data processing',
    long_description='',
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.16.0',
        'scipy>=1.2.0',
        'pandas>=0.24.0',
        'scikit-learn==0.21.3',
        'PyWavelets>=1.0.1',
        'matplotlib>=3.0.2',
        'dill>=0.2.7.1',
        'pint>=0.8.1',
        'tdigest>=0.5.2.2',
        'tqdm==4.30.0',
        'segyio==1.8.3',
        'scikit-image>=0.13.1',
        'numba>=0.35.0'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.12'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.12'],
        'keras': ['keras>=2.0.0'],
        'torch': ['torch>=1.0.0'],
        'hmmlearn': ['hmmlearn==0.2.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ],
)
