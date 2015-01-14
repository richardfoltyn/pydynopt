#!/bin/bash

VENV_ROOT=$HOME/virtualenv/hpc
PDO_ROOT=$HOME/repos/pydynopt/src

. $VENV_ROOT/bin/activate

SETUP="
from memviewtest import mvtest, mvtest2
from cython import double
import numpy as np
arr = np.ones((1000, 5, 1000, 11), dtype=np.float)
"

STMT="mvtest[double](arr, 10, 5)"
python3 -m timeit -s "${SETUP}" "${STMT}"

STMT="mvtest2[double](arr, 10, 5)"
python3 -m timeit -s "${SETUP}" "${STMT}"