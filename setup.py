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
    author='DATA Lab@Texas A&M University',
    author_email='khlai037@tamu.edu',
    url='https://tods-doc.github.io',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    package_data={
        'tods': ['resources/.entry_points.ini', 
                 'resources/.requirements.txt',
                 'resources/default_pipeline.json'
                 ]
    },
    install_requires=[
        'd3m',
        'Jinja2',
        'numpy==1.18.2',
        'simplejson==3.12.0',
        'scikit-learn==0.22.0',
	'statsmodels==0.11.1',
        'PyWavelets>=1.1.1',
        'pillow==7.1.2',
        'tensorflow', # should be removed later
        'keras', # should be removed later
        'pyod',
        'nimfa==1.4.0',
        'stumpy==1.4.0',
        'more-itertools==8.5.0',
    ],

    entry_points = merge_entry_points()

)

try:
    subprocess.run(['pip', 'install', '-r', 'tods/resources/.requirements.txt'])
except Exception as e:
    print(e)

