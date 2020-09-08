# Axolotl

This package provides an easy and high level abstraction 
of the [D3M](https://gitlab.com/datadrivendiscovery/d3m) API for AutoML. It contains a suit of basic
requirements and building blocks 
[primitives](https://gitlab.com/datadrivendiscovery/primitives).

## Installation

The package contains two different version of dependencies,
one with GPU support and other that uses CPU. For the installation
we strongly encourage the use of a python 3.6 virtual environment. 

* CPU version.
```bash
pip3 install -e git+https://gitlab.com/axolotl1/axolotl.git@devel#egg=axolotl[cpu]
```

* GPU version.
```bash
pip3 install -e git+https://gitlab.com/axolotl1/axolotl.git@devel#egg=axolotl[gpu]
```

Note:
For MacOs, pycurls needs to be manually installed:
```bash
PYCURL_SSL_LIBRARY=openssl LDFLAGS="-L/usr/local/opt/openssl/lib" CPPFLAGS="-I/usr/local/opt/openssl/include" pip install --no-cache-dir pycurl==7.43.0.3
```

## Usage
For new users we recommend installing the package and then cloning it via
```bash
git clone --recursive https://gitlab.com/axolotl1/axolotl.git
```

Then start jupyter lab via
```bash
jupyter lab
```
And then open the [examples](https://gitlab.com/axolotl1/axolotl/-/tree/devel/examples)
directory and try to run them.