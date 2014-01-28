from __future__ import absolute_import, print_function, division

import numpy as np

from ..common import ConvergenceError
from ..grid import DynoptResult
from ..grid import ProblemSpecExogenous


def _action_info(ps, i_state, ix_state=0):
    act_dummy = ps.actions(i_state)

    if isinstance(act_dummy, np.ndarray):
        dtype = act_dummy.dtype
    else:
        # TODO: more sophisticated type detection???
        dtype = type(act_dummy)

    if act_dummy.ndim == 1:
        dim_a = 1
    else:
        dim_a = act_dummy.shape[1]

    return dim_a, dtype


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


def pfi(ps, tol=1e-8, n_accel=30, maxiter=1000):

    do_exo = isinstance(ps, ProblemSpecExogenous)

    if do_exo:
        nstates_exo, nstates_end = ps.nstates_exo, ps.nstates_end
    else:
        nstates_exo, nstates_end = 1, ps.nstates

    v_beg = np.zeros((nstates_end, nstates_exo), dtype=float)
    v_end = np.empty_like(v_beg)
    v_old = np.empty_like(v_beg)
    util = np.empty_like(v_beg)
    idx_to = np.empty_like(v_beg, dtype=int)

    if do_exo:
        ix_next = np.ndarray(nstates_exo, dtype=object)
        px_next = np.empty_like(ix_next)

        # we neither know the length of next-period aggregate states, and the
        # number of possible states may vary across current states. Hence need to
        # use object type.
        for ix in range(nstates_exo):
            ix_next[ix], px_next[ix] = ps.transitions_exo(ix)
            assert np.all(np.min(px_next[ix]) >= 0)
            assert np.all(np.abs(np.sum(px_next[ix]) - 1) < 1e-12)
    else:
        ix_next = np.array(0).reshape((1, 1))
        px_next = np.array(1).reshape((1, 1))

    dim_a, dtype_a = _action_info(ps, 0, 0)
    opt_choice = np.empty((nstates_end, nstates_exo, dim_a), dtype=dtype_a)

    for it in range(maxiter):
        np.copyto(v_old, v_beg)

        for ix in range(nstates_exo):
            v_end[:, ix] = ps.discount * v_beg[:, ix_next[ix]].dot(px_next[ix])

            if divmod(it, n_accel)[1] == 0:
                for i in range(nstates_end):
                    act_try = ps.actions(i).reshape((-1, dim_a))

                    # TODO: include pnext in case we want to allow for linear
                    # interpolation
                    inext = ps.transitions(act_try, i, ix)
                    inext = inext.reshape((inext.shape[0], -1))
                    # assert np.all(np.abs(pnext.sum(axis=1) - 1) < 1e-12)

                    util_try = ps.util(act_try, i, ix)
                    # v_try = u + np.sum(vend_ix[inext] * pnext, axis=1)
                    v_try = util_try + v_end[inext, ix]
                    j_max = v_try.argmax()
                    opt_choice[i, ix, :] = act_try[j_max, :]

                    idx_to[i, ix] = inext[j_max]
                    util[i, ix] = util_try[j_max]

        # Update value fun, possibly without re-optimization.
        for ix in range(nstates_exo):
            v_beg[:, ix] = util[:, ix] + v_end[idx_to[:, ix], ix]

        if divmod(it, n_accel)[1] == 0:
            dv = np.max(np.abs(v_beg - v_old))
            print('pfi: Iteration %4d, delta V = %e' % (it + 1, dv))

            if dv < tol:
                if dim_a == 1:
                    opt_choice = opt_choice.squeeze(axis=(2,))
                res = DynoptResult(ps, v_beg, opt_choice, it, dv)
                return res
    else:
        raise ConvergenceError(it, dv)




