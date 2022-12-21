import os
from setuptools import setup, find_packages
import subprocess
import logging

PACKAGE_NAME = 'tods'

def read_file_entry_points(fname):
    with open(fname) as entry_points:
        return entry_points.read()

def merge_entry_points():
    entry_list = ['tods/resources/.entry_points.ini']
    merge_entry = []
    for entry_name in entry_list:
        entry_point = read_file_entry_points(entry_name).replace(' ', '')
        path_list = entry_point.split('\n')[1:]
        merge_entry += path_list
    entry_point_merge = dict()
    entry_point_merge['d3m.primitives'] = list(set(merge_entry)) # remove dumplicated elements
    return entry_point_merge

setup(
    name=PACKAGE_NAME,
    version='0.0.2',
    description='Automated Time-series Outlier Detection System',
    author='DATA Lab@Rice University',
    author_email='khlai037@rice.edu',
    url='https://tods-doc.github.io',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    package_data={
        'tods': ['resources/.entry_points.ini', 
                 'resources/.requirements.txt',
                 'resources/default_pipeline.json'
                 ]
    },
    install_requires=[
        'grpcio-testing==1.32.0',
        'tamu_d3m==2022.05.23',
        'tamu_axolotl',
        'numpy<=1.21.2',
        'combo',
        'simplejson==3.12.0',
        'scikit-learn',
	    'statsmodels==0.11.1',
        'PyWavelets>=1.1.1',
        'pillow==7.1.2',
        'tensorflow==2.4',
        'keras==2.4.0',
        'pyod==1.0.5',
        'nimfa==1.4.0',
        'stumpy==1.4.0',
        'more-itertools==8.5.0',
        'xgboost',
        'ray[tune]>=1.13.0'
    ],

    entry_points = merge_entry_points()

)
