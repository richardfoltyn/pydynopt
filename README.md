
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

# pydynopt

_pydynopt_ is a collection of functions intended to solve common computational 
and plotting tasks that arise when solving macroeconomic models in Python.

Author: Richard Foltyn

## Development setup

This repository uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. From a checkout, create the development environment and install all project, development, acceleration, and compression dependencies:

```bash
uv sync
```

Run project commands through `uv`:

```bash
uv run pytest -n auto
```

The development dependency group includes `numba` for accelerated implementations and `lz4` and `pyzstd` for compressed pickle persistence.

## License

Unless stated otherwise in individual files, this work is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
