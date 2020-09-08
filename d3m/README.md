# Common code for D3M project

This package provides a core package for D3M project with common code available.
It contains standard interfaces, reference implementations, and utility implementations.

## Installation

This package works with Python 3.6 and pip 19+. You need to have the following packages installed on the system (for Debian/Ubuntu):

* `libssl-dev`
* `libcurl4-openssl-dev`
* `libyaml-dev`

You can install latest stable version from [PyPI](https://pypi.org/):

```
$ pip3 install d3m
```

To install latest development version:

```
$ pip3 install -e git+https://gitlab.com/datadrivendiscovery/d3m.git@devel#egg=d3m
```

When cloning a repository, clone it recursively to get also git submodules:

```
$ git clone --recursive https://gitlab.com/datadrivendiscovery/d3m.git
```

## Changelog

See [HISTORY.md](./HISTORY.md) for summary of changes to this package.

## Documentation

Documentation for the package is available at [https://docs.datadrivendiscovery.org/](https://docs.datadrivendiscovery.org/).

## Contributing

See [CODE_STYLE.md](./CODE_STYLE.md) for our coding style and contribution guide. Please ensure any merge requests you open follow this guide.

## Repository structure

`master` branch contains latest stable release of the package.
`devel` branch is a staging branch for the next release.

Releases are [tagged](https://gitlab.com/datadrivendiscovery/d3m/tags).

## About Data Driven Discovery Program

DARPA Data Driven Discovery (D3M) Program is researching ways to get machines to build
machine learning pipelines automatically. It is split into three layers:
TA1 (primitives), TA2 (systems which combine primitives automatically into pipelines
and executes them), and TA3 (end-users interfaces).
