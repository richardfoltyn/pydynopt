__author__ = 'Richard Foltyn'

from pydynopt.compilers import njit


def vinterp(x0, x, fx, out):
    bnd = x.shape[0] - 1
    x_min, x_max = x[0], x[-1]
    for k in range(len(x0)):
        x0_k = x0[k]
        if x0_k < x_min:
            res = fx[0]
        elif x0_k > x_max:
            res = fx[-1]
        else:
            iu = 0

            xu = x[iu]
            while x0_k >= xu and iu < bnd:
                iu += 1
                xu = x[iu]
            il = iu - 1
            xl = x[il]
            fx_l = fx[il]
            res = fx_l + (fx[iu] - fx_l)/(xu - xl) * (x0_k - xl)
        out[k] = res

_sig = 'void(f8[:], f8[:], f8[:], f8[:])'
vinterp = njit(_sig)(vinterp)


def interp(x0, x, fx):
    bnd = x.shape[0] - 1
    x_min, x_max = x[0], x[-1]
    if x0 < x_min:
        res = fx[0]
    elif x0 > x_max:
        res = fx[-1]
    else:
        iu = 0

        xu = x[iu]
        while x0 >= xu and iu < bnd:
            iu += 1
            xu = x[iu]
        il = iu - 1
        xl = x[il]
        fx_l = fx[il]
        res = fx_l + (fx[iu] - fx_l)/(xu - xl) * (x0 - xl)
    return res

_sig = 'f8(f8, f8[:], f8[:])'
interp = njit(_sig)(interp)