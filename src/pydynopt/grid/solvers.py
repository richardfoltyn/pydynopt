from __future__ import absolute_import, print_function, division

import numpy as np
from numpy import ravel_multi_index as sub2ind, unravel_index as ind2sub
from numpy import vstack

from scipy.sparse import csr_matrix

from ..common import ConvergenceError
from ..grid import DynoptResult
from ..grid import ProblemSpecExogenous
from ..utils import cartesian_op

def _action_info(ps, i_state, ix_state=0):
    act_dummy, _, inext, pnext = ps.actions(i_state)

    if isinstance(act_dummy, np.ndarray):
        dtype_a = act_dummy.dtype
    else:
        dtype_a = type(act_dummy)

    if act_dummy.ndim == 0 or act_dummy.ndim == 1:
        dim_a = 1
    else:
        dim_a = act_dummy.shape[1]

    if inext.ndim == 1:
        inext_len = inext.shape[0]
    else:
        inext_len = inext.shape[1]

    # TODO: improve detection of inext dimension
    return dim_a, dtype_a, inext_len


def vfi(ps, tol=1e-8, maxiter=500):

    n_states = ps.nstates

    v = np.zeros(n_states, dtype=float)
    v_curr = np.empty(n_states, dtype=float)

    # determine dimensionality of action space (i.e. number of columns in
    # matrix returned by actions())
    act_dummy = ps.actions(0)
    if act_dummy.ndim == 1:
        dim_a = 1
    else:
        dim_a = act_dummy.shape[1]

    opt_choice = np.empty((n_states, dim_a))

    for it in range(maxiter):
        for i in range(n_states):
            # make sure we have a matrix of actions
            act = ps.actions(i).reshape((-1, dim_a))
            inext, pnext = ps.transitions(act, i)
            assert np.all(np.abs(pnext.sum(axis=1) - 1) < 1e-12)

            v_cont = np.sum(v[inext] * pnext, axis=1)
            v_try = ps.util(act, i).ravel() + ps.discount * v_cont

            j_max = v_try.argmax()
            opt_choice[i, :], v_curr[i] = act[j_max, :], v_try[j_max]

        dv = np.abs(v - v_curr).max()
        v, v_curr = v_curr, np.empty(n_states, dtype=float)

        if divmod(it + 1, 10)[1] == 0:
            print('vfi: Iteration %d, dv=%e' % (it + 1, dv))

        if dv < tol:
            if act_dummy.ndim == 1:
                opt_choice.squeeze()
            res = DynoptResult(v, opt_choice, it, dv)
            return res
    else:
        raise ConvergenceError(it, dv)


def pfi(ps, tol=1e-8, n_accel=30, maxiter=1000, transm=True):
    assert isinstance(ps, ProblemSpecExogenous)

    n_x, n_n, n = ps.nstates_exo, ps.nstates_end, ps.nstates
    beta = ps.discount

    v_beg = np.zeros((n_x, n_n), dtype=np.float64)
    v_end = np.empty_like(v_beg)
    v_old = np.empty_like(v_beg)
    util_cont = np.empty_like(v_beg)

    # Arrays to hold next-period states and probabilities of exogenous
    # transitions. Let's home there are not too many exogenous states,
    # otherwise this will blow up your memory. :)
    pnext_x = np.zeros((n_x, n_x), dtype=np.float64)

    for ix in range(n_x):
        lidx, lprob = ps.transitions_exo(ix)
        assert np.all(np.min(lprob) >= 0)
        assert np.all(np.abs(np.sum(lprob) - 1) < 1e-12)
        pnext_x[ix, lidx] = lprob

    dim_a, dtype_a, dim_inext = _action_info(ps, 0, 0)
    opt_choice = np.empty((n_x, n_n, dim_a), dtype=dtype_a)

    # Arrays to hold next-period states and probabilities of endogenous
    # transitions implied by optimally chosen action.
    # TODO: Currently we assume that each action results in transition
    # possibilities to the same number of states, which is somewhat
    # unflexible.
    inext_end = np.empty((n_x, n_n, dim_inext), dtype=np.uint32)
    pnext_end = np.empty((n_x, n_n, dim_inext),
                         dtype=np.float64)

    for it in range(maxiter):
        # np.copyto(v_old, v_beg)
        v_old, v_beg = v_beg, v_old

        for ix in range(n_x):
            v_end[ix] = beta * pnext_x[ix].dot(v_old)

            if it % n_accel == 0:
                for i in range(n_n):
                    # the return values from actions() must be such that each
                    # row represents an action. acts can be a 1d-array,
                    # but inext and pnext must be matrices.
                    acts, u_cont, inext, pnext = ps.actions(i, ix)

                    assert np.all(np.abs(pnext.sum(axis=-1) - 1) < 1e-12)

                    v_try = u_cont + np.sum(v_end[ix, inext] * pnext, axis=-1)

                    j_max = v_try.argmax()
                    opt_choice[ix, i, :] = acts[j_max]

                    inext_end[ix, i] = inext[j_max]
                    pnext_end[ix, i] = pnext[j_max]
                    util_cont[ix, i] = u_cont[j_max]

        # Update value fun, possibly without re-optimization.
        for ix in range(n_x):
            v_ix = v_end[ix, inext_end[ix]]
            v_beg[ix] = util_cont[ix] + np.sum(v_ix * pnext_end[ix], axis=-1)

        if it % n_accel == 0:
            dv = np.max(np.abs(v_beg - v_old))

            if dv < tol:
                if transm:
                    # Construct a nstates x nstates sparse matrix where
                    # m[i,j] = Prob[ transition to state j | state i],
                    # where i, j are linear indices in 0,1,..., nstates-1

                    # Pre-allocate some memory to contain row indices, column
                    # indices and transition probabilities:
                    arr_size = n_x * n_n * dim_inext * \
                               np.sum(pnext_x[0] > 0)
                    probs = np.empty((arr_size, ), dtype=float)
                    indptr = np.empty((n + 1, ), dtype=int)
                    indices = np.empty_like(probs, dtype=int)
                    indptr[0] = 0
                    # build sparse transition matrix in CSR format
                    for i in range(ps.nstates_end):
                        for ix in range(ps.nstates_exo):
                            iend = vstack(ind2sub(inext_end[ix, i],
                                                  dims=ps.grid_shape_end))
                            inext_ix = np.nonzero(pnext_x[ix])[0]
                            iexo = vstack(ind2sub(inext_ix, dims=ps.grid_shape_exo))

                            cart = cartesian_op((iend, iexo), dtype=iend.dtype)
                            inext = sub2ind(tuple(cart), dims=ps.grid_shape)

                            pnext = cartesian_op((pnext_end[ix, i], pnext_x[ix]),
                                                 op=np.prod).squeeze()

                            last_idx = i * ps.nstates_exo + ix
                            curr_size = len(probs)
                            if indptr[last_idx] + len(inext) > curr_size:
                                new_size = (curr_size * n) // last_idx
                                probs = np.resize(probs, (new_size, ))
                                indices = np.resize(indices, (new_size, ))

                            indptr[last_idx + 1] = indptr[last_idx] + len(inext)
                            indices[indptr[last_idx]:indptr[last_idx + 1]] = inext
                            probs[indptr[last_idx]:indptr[last_idx + 1]] = pnext

                    max_idx = indptr[n]
                    transm = csr_matrix(
                        (probs[:max_idx], indices[:max_idx], indptr),
                        shape=(n, n))

                    assert transm.min() >= 0
                    assert np.all(np.abs(transm.sum(axis=1) - 1) < 1e-12)
                else:
                    transm = None

                shp = ps.grid_shape
                v = np.ascontiguousarray(v_beg.swapaxes(0, 1)).reshape(shp)
                # care must be taken when reshaping into array where
                # endogenous dimensions appear before exogenous ones,
                # as the reshaping must not affect action dimension! (so a
                # simple transpose will mix up actions)
                oc = np.ascontiguousarray(opt_choice.swapaxes(0, 1))
                oc = oc.reshape(shp + (-1,))

                res = DynoptResult(ps, v, oc, it, dv, transm=transm)

                return res
    else:
        raise ConvergenceError(it, dv)