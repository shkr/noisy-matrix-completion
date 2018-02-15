import numpy as np
import scipy as sp

MAX_ITERATIONS = 1000


class MatrixDecomposition(object):

    def __init__(self,
                 Y_obs,
                 mask=None,
                 lam=0.025,
                 mu=0.005,
                 alpha=10000,
                 tolerance=1e-2,
                 max_iterations=MAX_ITERATIONS):
        """
        Decomposes an optionally incomplete matrix, into two components, one low rank, and one sparse.
        Also described here https://github.com/fela/matrix-completion

        :param Y_obs: The matrix to decompose.
        It is supposed that Y is equal to TH + GA + W,
        where TH is an approximately low rank matrix,
        GA is a sparse "spiky" matrix, and W is noise.
        :param mask: The elements in the Y_obs matrix that
        are unknown
        :param lambda_d: regularization parameter for the low rank TH matrix.
        :param mu_d: regularization parameter for the sparse GA matrix. Use higher values if no spikes are expected.
        :param alpha: parameter that limits the maximum element of the low rank TH matrix.
        :param max_iterations: The number of proximal iterations
        """
        self.Y_obs = Y_obs
        # mask is optional
        if mask is None:
            mask = (Y_obs * 0) + 1
        self.mask = mask
        self.lam = lam
        self.mu = mu
        self.alpha = alpha
        print("Maximum value permissible in the complete matrix:", alpha / sp.sqrt(sp.array(Y_obs.shape).prod()))
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.__set_model__()

    def __set_model__(self):
        self.theta = self.Y_obs * 0
        self.gamma = self.Y_obs * 0

    def train(self, n_iter):
        """
        Inference Loop
        Assuming Observation Operator = I
        A = I
        :return:
        """
        # Apply mask if not already applied
        mask = self.mask
        Y = np.where(mask, self.Y_obs, 0)
        lam = self.lam
        mu = self.mu
        alpha = self.alpha
        tolerance = self.tolerance

        # Initialization Values
        L = 2.0

        # Set Index Pointer
        iter_number = 0
        t_curr = 1.0

        # Set Pointers
        # 2 Calculate next theta, gamma
        theta_curr = theta_prev = self.theta
        gamma_curr = gamma_prev = self.gamma

        while iter_number < n_iter:

            # Calculate next t
            t_next = (1.0 + sp.sqrt(1 + 4 * t_curr ** 2)) / 2.0
            # calculate next Z & C from curr t, theta and gamma
            Z_next = theta_curr + ((t_curr - 1)/t_next)*(theta_curr-theta_prev)
            C_next = gamma_curr + ((t_curr - 1)/t_next)*(gamma_curr-gamma_prev)

            # Next
            # set next t, theta and gamma
            t_curr = t_next
            iter_number = iter_number + 1
            theta_prev = theta_curr
            gamma_prev = gamma_curr

            f = ((Z_next + C_next) - Y) * (1. / L)
            Z = Z_next - np.where(mask, f, 0)
            C = C_next - np.where(mask, f, 0)

            # sequence is initialized with Z_curr and C_curr
            # update theta_curr & gamma_curr
            theta_curr, gamma_curr = self.dykstra_like_proximal_algorithm(Z, C, lam, mu, alpha, L)

            mae = round(np.max(np.where(mask, abs((theta_curr + gamma_curr) - Y), 0)), 3)
            rmse = round(np.sqrt((np.where(mask, (theta_curr + gamma_curr) - Y, 0)**2).sum()/mask.sum()), 3)
            nuclear_norm = round(np.abs(sp.linalg.svd(theta_curr)[1]).sum(), 3)
            any_negative = (theta_curr < 0).sum()

            diff = round(sp.sqrt(((theta_curr - theta_prev) ** 2).sum()) + \
                   sp.sqrt(((gamma_curr - gamma_prev) ** 2).sum()), 3)

            break_condition = diff < tolerance

            if (iter_number + 1) % 100 == 0 or break_condition:
                print('{} iterations completed and current mae = {} rmse = {} and rmse wrt convergence = {}'.format(iter_number + 1, mae, rmse, diff))
                print('Nuclear Norm: {}, any_negative: {}'.format(nuclear_norm, any_negative))

            if break_condition:
                break

        self.theta, self.gamma = theta_curr, gamma_curr


    @classmethod
    def dykstra_like_proximal_algorithm(cls, Z, C, lam, mu, alpha, L):
        """
        prox_(lambda_mu_G)(Z) ~ W_n until convergence
        :param Z:
        :param tolerance:
        :return:
        """
        # Set Pointers
        iter_number = 0
        X = Z
        P = Q = Z * 0

        while iter_number < MAX_ITERATIONS:
            # Calculate Y_curr from X_curr and P_curr
            Y = cls.prox1(X + P, lam/L)
            # Now calculate P_next from X_curr, Y_curr
            P = X + P - Y
            # Now calculate X_next from Y_curr and Q_curr
            X_prev = X
            X = cls.prox2(Y + Q, alpha)
            # Now calculate Q_next from Y_curr, Q_curr and X_next
            Q = Y + Q - X

            if sp.sqrt(((X - X_prev)**2).sum()) < 1e-2:
                break

            # move forward
            iter_number = iter_number + 1

        return X, cls.soft_threshold(C, mu/L)

    @classmethod
    def prox1(cls, X, v1):
        """
        :param X:
        :param v1:
        :return:
        """
        U, s, Vh = sp.linalg.svd(X)
        d1, d2 = X.shape
        E = sp.linalg.diagsvd(s, d1, d2)
        S = cls.soft_threshold(E, v1)
        return U.dot(S.dot(Vh))

    @classmethod
    def soft_threshold(cls, X, s):
        """
        Soft Threshold
        :param X:
        :param s:
        :return:
        """
        return (X-s)*(X>s) + (X+s)*(X<-s)

    @classmethod
    def prox2(cls, X, alpha):
        """
        Projection in Q
        :param X:
        :param alpha:
        :return:
        """
        limit = alpha / sp.sqrt(sp.array(X.shape).prod())
        X = sp.minimum(X, limit)
        X = sp.maximum(X, -limit)
        return X