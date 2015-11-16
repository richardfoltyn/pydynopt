
import numpy as np

def reshape_result(biter, mv_flat):
    """
    Create result array or scalar depending on dimension of inputs
    """

    if biter.nd == 0:
        # return scalar
        out = mv_flat[0]
    elif biter.nd == 1:
        # return 1-d array, no reshaping required
        out = np.asarray(mv_flat)
    else:
        # return n-dimensional array
        out = np.asarray(mv_flat).reshape(biter.shape)

    return out