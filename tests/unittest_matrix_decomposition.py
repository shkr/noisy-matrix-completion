# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:18:35 2012

@author: Fela Winkelmolen
"""

import unittest
from noisy_matrix_completion.matrix_decomposition import matrix_decomposition
import scipy as sp
sp.random.seed(0)
from noisy_matrix_completion.matrix_generation import  low_rank, spiky, selection


class TestMatrixGeneration(unittest.TestCase):

    def test_print_problem(self):
        size = 30
        matrix_rank = 5
        perc_masked = 0.2


        TH = low_rank(size, matrix_rank)
        GA = spiky(size)

        X = selection(size, perc_masked)
        Y_obs = (TH + GA) * X
        A, B = matrix_decomposition(Y_obs, mask=X)

        self.assertTrue(True, 'TH: ' + str(TH[1, :]))
        self.assertTrue(True, 'GA: ' + str(GA[1, :]))
        self.assertTrue(True, 'Y_obs: ' + str(Y_obs[1, :]))

        self.assertTrue(True, '~A: ' + str(A[1, :]))
        self.assertTrue(True, '~B: ' + str(B[1, :]))

    def test_average_error_less_than_threshold(self):
        size = 30
        matrix_rank = 5
        perc_masked = 0.2
        average_error_threshold = 0.2

        TH = low_rank(size, matrix_rank)
        GA = spiky(size)

        X = selection(size, perc_masked)
        Y_obs = (TH + GA) * X
        A, B = matrix_decomposition(Y_obs, mask=X)
        self.assertLess(
            (abs(TH - A) / abs(TH)).sum() / (size ** 2),
            0.2,
            "The average error must be less than {}".format(average_error_threshold)
        )

    def test_maximum_spike_error_less_than_maximum_spike(self):
        size = 30
        matrix_rank = 5
        perc_masked = 0.2

        TH = low_rank(size, matrix_rank)
        GA = spiky(size)

        X = selection(size, perc_masked)
        Y_obs = (TH + GA) * X
        A, B = matrix_decomposition(Y_obs, mask=X)
        self.assertLess(
            (abs(GA - B)).max(),
            GA.max(),
            "The maximum spike error must be less than the maximum spike which is {}".format(GA.max())
        )


if __name__ == '__main__':
    unittest.main()
