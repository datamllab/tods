#!/usr/bin/env python3

import pkg_resources

package = pkg_resources.working_set.by_key['d3m']

oldest_dependencies = []

for requirement in package.requires():
    dependency = requirement.project_name
    if requirement.extras:
        dependency += '[' + ','.join(requirement.extras) + ']'
    for comparator, version in requirement.specs:
        if comparator == '==':
            if len(requirement.specs) != 1:
                raise ValueError('Invalid dependency: {requirement}'.format(requirement=requirement))
            dependency += '==' + version
        elif comparator == '<=':
            if len(requirement.specs) != 2:
                raise ValueError('Invalid dependency: {requirement}'.format(requirement=requirement))
        elif comparator == '>=':
            dependency += '==' + version

    oldest_dependencies.append(dependency)

for dependency in oldest_dependencies:
    print(dependency)
