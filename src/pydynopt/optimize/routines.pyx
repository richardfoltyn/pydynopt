
from libc.math cimport sqrt, fabs

cdef inline int sign(double x) nogil:
    if x < 0.0:
        return -1
    elif x > 0.0:
        return 1
    else:
        return 0


cpdef double fminbound(Optimizer opt, double x1, double x2, OptResult res,
                       double xatol=1e-5, unsigned int maxiter=500) nogil:

    cdef unsigned int maxfun = maxiter

    cdef int flag = 0

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))

    cdef double a = x1, b = x2

    cdef double fulc, nfc, xf, fx, e, rat, x
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = opt.objective(x)

    cdef unsigned int num
    num = 1

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * fabs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    cdef int si

    while fabs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = 1
        # Check for parabolic fit
        if fabs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = fabs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((fabs(p) < fabs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden section step
                golden = 1

        if golden:  # Do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = sign(rat) + (rat == 0)
        x = xf + si * max(fabs(rat), tol1)
        fu = opt.objective(x)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * fabs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    res.flag = flag
    res.fx_opt = fx
    res.x_opt = xf

    return xf