
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

# pydynopt

_pydynopt_ is a collection of functions intended to solve common computational 
and plotting tasks that arise when solving macroeconomic models in Python.

Author: Richard Foltyn

## Installation

To install `pydynopt` with base dependencies:

```bash
pip install .
```

The base installation omits optional acceleration and compression packages. You can request optional functionality using the following extras:

- **Numba acceleration**:
  ```bash
  pip install "pydynopt[numba]"
  ```
  Installs `numba` to accelerate computations. If not installed, `pydynopt` automatically falls back to pure-Python implementations.
- **Compression**:
  ```bash
  pip install "pydynopt[compression]"
  ```
  Installs `lz4` and `pyzstd` to support LZ4 and Zstandard formats for compressed pickle persistence. (Note that Python 3.14+ has standard-library Zstandard support).

To set up the development environment, run `uv sync` to install all optional and development packages via the `dev` dependency group.

## License

Unless stated otherwise in individual files, this work is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
