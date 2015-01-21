__author__ = 'Richard Foltyn'

import unittest2 as ut
import numpy as np

from pydynopt.processes.tauchen import tauchen


class TauchenTest(ut.TestCase):

    def test_tauchen(self):
        # Illustrate with the parametrization / results given in Tauchen (1986),
        # Table 1
        params = [[0.1, 0.101, 9],
                  [0.8, 0.167, 9],
                  [0.9, 0.229, 9],
                  [0.9, 0.229, 5]]

        # Results from Table 1 for implied lambda and sigma_y of discretized process
        tbl1_res = [[0.100, 0.103],
                    [0.798, 0.176],
                    [0.898, 0.253],
                    [0.932, 0.291]]

        results = list()

        for args in params:
            allargs = args + [3, False, True]
            results.append(tauchen(*allargs))

        fmt = '{:>2s}{:>10s}{:>10s}{:>15s}{:>10s}'
        header = fmt.format('N', 'lambda', 'sigma_y', 'lambda_bar', 'sigma_yab')

        print(header)
        print('-' * len(header))
        fmt = '{:2d}{:-10.2f}{:-10.3f}{:-15.3f}{:10.3f}'
        for i, res in enumerate(results):
            print(fmt.format(params[i][2], params[i][0], params[i][1], res[3],
                             res[4]))

            self.assertTrue(np.abs(tbl1_res[i][0] - res[3]) < 1e-3)
            self.assertTrue(np.abs(tbl1_res[i][1] - res[4]) < 1e-3)
