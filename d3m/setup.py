import os
import os.path
import sys
from setuptools import setup, find_packages

PACKAGE_NAME = 'd3m'
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
description = read_package_variable('__description__')
author = read_package_variable('__author__')

setup(
    name=PACKAGE_NAME,
    version=version,
    description=version,
    author=author,
    packages=find_packages(exclude=['contrib', 'docs', 'site', 'tests*']),
    package_data={'d3m': ['metadata/schemas/*/*.json', 'contrib/pipelines/*']},
    data_files=[('./', ['./entry_points.ini'])],
    install_requires=[
        'scikit-learn[alldeps]>=0.20.3,<=0.22.2.post1',
        'pytypes==1.0b5',
        'frozendict==1.2',
        'numpy>=1.15.4,<=1.18.2',
        'jsonschema>=3.0.2,<=3.2.0',
        'requests>=2.19.1,<=2.23.0',
        'strict-rfc3339==0.7',
        'rfc3987==1.3.8',
        'webcolors>=1.8.1,<=1.11.1',
        'dateparser>=0.7.0,<=0.7.2',
        'python-dateutil==2.8.1',
        'pandas>=0.23.4,<=1.0.3',
        'typing-inspect==0.5.0',
        'GitPython==3.1.0',
        'jsonpath-ng==1.4.3',
        'custom-inherit>=2.2.0,<=2.2.2',
        'PyYAML>=5.1,<=5.3',
        'pycurl>=7.43.0.2,<=7.43.0.5',
        'pyarrow>=0.15.1,<=0.16.0',
        'gputil>=1.3.0,<=1.4.0',
        'pyrsistent>=0.14.11,<=0.15.7',
        'scipy>=1.2.1,<=1.4.1',
        'openml==0.10.1',
    ],
    tests_require=[
        'asv==0.3.1',
        'docker[tls]==2.7',
    ],
    entry_points=read_entry_points(),
    url='https://gitlab.com/datadrivendiscovery/d3m',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    classifiers=[
          'License :: OSI Approved :: Apache Software License',
    ],
)
