import os
import sys
from setuptools import setup, find_packages

PACKAGE_NAME = 'common_primitives'
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


def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()


def read_entry_points():
    with open('entry_points.ini') as entry_points:
        return entry_points.read()


check_python_version()
version = read_package_variable('__version__')

setup(
    name=PACKAGE_NAME,
    version=version,
    description='D3M common primitives',
    author=read_package_variable('__author__'),
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    data_files=[('./', ['./entry_points.ini'])],
    install_requires=[
        'd3m',
        'pandas',
        'scikit-learn',
        'numpy',
        'lightgbm>=2.2.2,<=2.3.0',
        'opencv-python-headless<=4.1.1.26,>=4.1',
        'imageio>=2.3.0,<=2.6.0',
        'pillow==6.2.1',
        'xgboost>=0.81,<=0.90',
    ],
    entry_points=read_entry_points(),
    url='https://gitlab.com/datadrivendiscovery/common-primitives',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    classifiers=[
          'License :: OSI Approved :: Apache Software License',
    ],
)
