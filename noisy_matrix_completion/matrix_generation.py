import scipy as sp
import scipy.linalg as linalg
import numpy.random as random


def orthonormal(m):
    """
    calculate eigen vectors of a random symmetric m by m matrix
    and fetch its eigenvectors

    verify it it is orthonormal, recurse it tests fail
    :param m:
    :return:
    """
    A = sp.rand(m, m)
    S = A * A.T
    _, P = linalg.eigh(S)

    # verification
    I = sp.eye(m)
    tolerance = sp.finfo(float).eps * 16
    if (P.dot(P.T) - I > tolerance).any():
        return orthonormal(m)

    return P


def low_rank(m, rank):
    """
    Calculates a low rank matrix
    :param m:
    :param rank:
    :return:
    """
    diag = sp.concatenate((sp.rand(rank)*5+1, sp.zeros(m-rank)))
    E = sp.diag(diag)
    return orthonormal(m).dot(E).dot(orthonormal(m).T)


def spiky(m, spikes=None, avg=1.0, sigma=None):
    """
    Calculates a sparse-like and spiky matrix
    :param m:
    :param spikes:
    :param avg:
    :param sigma:
    :return:
    """
    # substitute None with default values
    spikes = spikes or m
    sigma = sigma or avg/6.0
    
    GA = sp.zeros((m, m))
    r = random.randint(0, m, spikes) # row indexes of spikes
    c = random.randint(0, m, spikes) # column indexes of spikes
    GA[[r, c]] = sigma * random.randn(spikes) + avg
    return GA


def selection(rows, columns, perc):
    """
    Returns a m x m U(0, 1) distributed
    matrix with perc% values set as False
    and others True
    :param m:
    :param perc:
    :return:
    """
    return sp.rand(rows, columns) >= perc
