# Common D3M primitives

A common set of primitives for D3M project, maintained together.
It contains example primitives, various glue primitives, and other primitives performers
contributed.

## Installation

This package works on Python 3.6+ and pip 19+.

This package additional dependencies which are specified in primitives' metadata,
but if you are manually installing the package, you have to first run, for Ubuntu:

```
$ apt-get install build-essential libopenblas-dev libcap-dev ffmpeg
$ pip3 install python-prctl
```

To install common primitives from inside a cloned repository, run:

```
$ pip3 install -e .
```

When cloning a repository, clone it recursively to get also git submodules:

```
$ git clone --recursive https://gitlab.com/datadrivendiscovery/common-primitives.git
```

## Changelog

See [HISTORY.md](./HISTORY.md) for summary of changes to this package.

## Repository structure

`master` branch contains latest code of common primitives made against the latest stable
release of the [`d3m` core package](https://gitlab.com/datadrivendiscovery/d3m) (its `master` branch).
`devel` branch contains latest code of common primitives made against the
future release of the `d3m` core package (its `devel` branch).

Releases are [tagged](https://gitlab.com/datadrivendiscovery/d3m/tags) but they are not done
regularly. Each primitive has its own versions as well, which are not related to package versions.
Generally is the best to just use the latest code available in `master` or `devel`
branches (depending which version of the core package you are using).

## Testing locally

For each commit to this repository, tests run automatically in the
[GitLab CI](https://gitlab.com/datadrivendiscovery/common-primitives/pipelines). 

If you don't want to wait for the GitLab CI test results and run the tests locally,
you can install and use the [GitLab runner](https://docs.gitlab.com/runner/install/) in your system.

With the local GitLab runner, you can run the tests defined in the [.gitlab-ci.yml](.gitlab-ci.yml)
file of this repository, such as:

```
$ gitlab-runner exec docker style_check
$ gitlab-runner exec docker type_check
```

You can also just try to run tests available under `/tests` by running:

```
$ python3 run_tests.py
```

## Contribute

Feel free to contribute more primitives to this repository. The idea is that we build
a common set of primitives which can help both as an example, but also to have shared
maintenance of some primitives, especially glue primitives.

All primitives are written in Python 3 and are type checked using
[mypy](http://www.mypy-lang.org/), so typing annotations are required.

## About Data Driven Discovery Program

DARPA Data Driven Discovery (D3M) Program is researching ways to get machines to build
machine learning pipelines automatically. It is split into three layers:
TA1 (primitives), TA2 (systems which combine primitives automatically into pipelines
and executes them), and TA3 (end-users interfaces).
