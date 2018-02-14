from matrix_generation import *
import matrix_decomposition as md
import numpy as np
import matplotlib.pyplot as plt


class RealDataExperiment:

    def __init__(self, TH, GA, W, missing_pct=.2):
        """
        Experiment to run the noisy matrix completion
        approximation algorithm
        :param TH:
        :param GA:
        :param W:
        :param missing_pct:
        """
        self.missing_pct = missing_pct
        self.TH, self.GA, self.W = TH, GA, W
        self.init_arguments()

    def run(self):
        """
        Runs the matrix completion routine
        and calculates the RMSD on the missing
        values
        :return:
        """
        TH, GA = self.TH, self.GA
        mask = self.mask
        A, B = md.matrix_decomposition(self.Y_obs, self.mask)
        
        n = (1-mask).sum()
        diff = (TH-A)*(1-mask)
        sqerr = (diff**2).sum()
        if n != 0:
            sqerr /= n
        return np.sqrt(sqerr)
    
    def init_arguments(self):
        self.Y = self.TH + self.GA + self.W
        rows, cols = self.Y.shape
        self.mask = selection(rows, cols, self.missing_pct)
        self.Y_obs = self.Y * self.mask


class SyntheticDataExperiment(object):

    def __init__(self, size=30, rank=3, missing_pct=0.2, noise=5.):
        # default values
        self._size = size
        self._rank = rank
        self._missing_pct = missing_pct
        self._noise = noise
        self._generate_data()

    def _generate_data(self):
        """
        sets the following instance variables:
        * _Y is the input matrix of the decomposition algorithm
        * _Mask is the mask of _Y equal to True in the positions
          where _Y is observed
        * _TH is the first component of _Y, which we want to recover
        * _GA is the other component
         """
        size = self._size
        TH = low_rank(size, self._rank) + sp.rand(size, size) / 50 * self._noise
        GA = spiky(size)
        Mask = selection(size, size, self._missing_pct)
        Y = (TH + GA) * Mask
        self._Y, self._TH, self._GA, self._Mask = Y, TH, GA, Mask

    def run(self):
        """
        Runs the algo on a synthetic dataset
        :return:
        """
        TH, GA = self._TH, self._GA
        Y, Mask = self._Y, self._Mask
        A, B = md.matrix_decomposition(Y, Mask)
        error = ((TH - A) ** 2).sum() / TH.size
        print "mean square error:", error
        error2 = ((TH - Y * Mask) ** 2).sum() / TH.size
        print "mse for naive solution:", error2
        print "improvement:", error2 / error, "times"


def load_data(filename):
    M = [[int(i) for i in line.split(',')] for line in open(filename)]
    M = sp.array(M)
    y = M[:, 0]
    X = M[:, range(1, M.shape[1])]
    return X, y


def missing_data_experiment(TH, GA, W, *missing_pcts):
    """
    mu_d and lambda_d experiment pending
    :param steps:
    :param runs:
    :param TH:
    :param GA:
    :param W:
    :return:
    """
    runs = 5
    y = []
    x = []
    for pct in missing_pcts:
        print('Running for {} missing data pct'.format(pct))
        y.append(pct)
        acc = []
        e = RealDataExperiment(TH, GA, W, missing_pct=pct)
        for i in range(runs):
            e.init_arguments()
            acc.append(e.run())
        curr_acc = sp.array(acc).mean()
        print('Missing data pct: {} / RMSD error: {} \n'.format(pct, curr_acc))
        x.append(curr_acc)
    return y, x