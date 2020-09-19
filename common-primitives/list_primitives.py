#!/usr/bin/env python3

import argparse
import configparser
import re


class CaseSensitiveConfigParser(configparser.ConfigParser):
    optionxform = staticmethod(str)


parser = argparse.ArgumentParser(description='List enabled common primitives.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--suffix', action='store_true', help='list primitive suffixes of all enabled common primitives')
group.add_argument('--python', action='store_true', help='list Python paths of all enabled common primitives')
group.add_argument('--files', action='store_true', help='list file paths of all enabled common primitives')

args = parser.parse_args()

entry_points = CaseSensitiveConfigParser()
entry_points.read('entry_points.ini')

for primitive_suffix, primitive_path in entry_points.items('d3m.primitives'):
    if args.python:
        print("d3m.primitives.{primitive_suffix}".format(primitive_suffix=primitive_suffix))
    elif args.suffix:
        print(primitive_suffix)
    elif args.files:
        primitive_path = re.sub(':.+$', '', primitive_path)
        primitive_path = re.sub('\.', '/', primitive_path)
        print("{primitive_path}.py".format(primitive_path=primitive_path))

