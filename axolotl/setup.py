import os
import os.path
import sys
from setuptools import setup, find_packages
import subprocess

PACKAGE_NAME = 'axolotl'
MINIMUM_PYTHON_VERSION = 3, 6


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    raise KeyError("'{0}' not found in '{1}'".format(key, module_path))


check_python_version()
version = read_package_variable('__version__')
description = read_package_variable('__description__')
setup(
    name=PACKAGE_NAME,
    version=version,
    description=version,

    packages=find_packages(exclude=['tests*']),
    license='Apache-2.0',
    classifiers=[
          'License :: OSI Approved :: Apache Software License',
    ],
    install_requires=[
        'd3m',
        'grpcio',
        'grpcio-tools',
        'grpcio-testing',
        'ray',
        'networkx',
    ],
    extras_require={
        'cpu': ['tensorflow==2.2.0'],
        'gpu': ['tensorflow-gpu==2.2.0']
    }
)
