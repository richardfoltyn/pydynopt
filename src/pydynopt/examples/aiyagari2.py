from __future__ import absolute_import, print_function, division
from math import log

import numpy as np

from pydynopt.grid import *
from pydynopt.utils import rouwenhorst, normprob
from numpy import unravel_index as ind2sub
from numpy import ravel_multi_index as sub2ind

import matplotlib.pyplot as plt


class AiyagariPE2(ProblemSpecExogenous):

    def __init__(self, a_ngrid=450, a_max=20, r=0.04, kshare=.36, A=1,
                 gamma=2, b=0,
                 ky=3, eps_ngrid=5, eps_mu=0, eps_rho=.9, eps_sigma=.03):

        self.r = r
        self.alpha = kshare
        self.delta = kshare * (A ** 2) / ky - r
        self.gamma = gamma
        self.b = b
        self.beta = 1/(1+r)

        self.ln_eps_state, self.eps_trans = rouwenhorst(eps_ngrid, eps_mu,
                                                        eps_rho, eps_sigma)

        tmp = np.linalg.matrix_power(self.eps_trans, 1000)
        self.eps_probst = normprob(tmp[1, :])

        # normalize state space such that the aggregate labor supply given
        # the stationary distribution of productivities is 1
        self.grid_eps = np.exp(self.ln_eps_state) / \
                        self.eps_probst.dot(np.exp(self.ln_eps_state))

        # capital-labor ratio and wage rate implied by r
        kl = ((A * kshare)/(r + self.delta)) ** (1/(1-kshare))
        self.wage = A * (1-kshare) * kl ** kshare

        # determine 'natural' borrowing constraint:
        if r <= 0:
            self.phi = self.b
        else:
            self.phi = min(self.b, self.wage * self.grid_eps[0]/r)

        # create asset grid such that it is finer for smaller asset values
        ln_a = np.linspace(0, log(a_max + self.phi + 1), a_ngrid)
        self.grid_a = np.exp(ln_a) - 1 - self.phi

        super(AiyagariPE2, self).__init__(grid_shape_end=self.grid_a.shape,
                                          grid_shape_exo=self.grid_eps.shape,
                                          discount=self.beta)

    def run(self):
        result = pfi(self, tol=1e-5)
        print('Completed after %d iterations (tol=%e)' %
              (result.iterations, result.tol))

        return result

    def actions(self, i_state, ix_state=0):
        ia, ie = i_state, ix_state

        a_beg = self.grid_a[ia]
        a_max = a_beg * (1 + self.r) + self.wage * self.grid_eps[ie]

        # return indices of viable next-period asset positions as actions
        # Note that nonzero returns a tuple of indices.
        return np.nonzero(self.grid_a <= a_max)[0]

    def transitions(self, actions, i_state, ix_state=0):
        return actions

    def transitions_exo(self, ix_state=0):
        ie = ix_state

        iEnext = np.nonzero(self.eps_trans[ie, :])[0]
        pEnext = self.eps_trans[ie, iEnext]

        return iEnext, pEnext

    def util(self, actions, i_state, ix_state=0):
        ia, ie = i_state, ix_state

        cons = (1+self.r) * self.grid_a[ia] + self.wage * self.grid_eps[ie] -\
               self.grid_a[actions]

        u = (cons ** (1-self.gamma) - 1)/(1-self.gamma)
        return u


def main():
    a_ngrid = 450
    eps_ngrid = 5
    ps = AiyagariPE2(a_ngrid=a_ngrid, eps_ngrid=eps_ngrid)
    res = ps.run()

    v = res.vfun.reshape((a_ngrid, -1))
    opt_choice = res.opt_choice.reshape((a_ngrid, -1))

    fig, axes = plt.subplots(2, 1, sharex=True)
    for i in range(0, eps_ngrid, 2):
        axes[0].plot(ps.grid_a, v[:, i], lw=2,
                     label=r'$\epsilon=%4.3f$' % ps.grid_eps[i],
                     color=plt.cm.jet(i / eps_ngrid))

        axes[1].plot(ps.grid_a, ps.grid_a[opt_choice[:, i]], lw=2,
                     label=r'$\epsilon=%4.3f$' % ps.grid_eps[i],
                     color=plt.cm.jet(i / eps_ngrid))

    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    plt.show()

    return res



if __name__ == "__main__": main()