import numpy as np


class MatrixALS:
    def __init__(self,
                 obs,
                 n_factors,
                 x_reg,
                 y_reg,
                 random_state=None,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix using ALS algorithm

        Params
        ======
        obs : (ndarray)
            n x m incomplete matrix

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model

        x_reg : (float)
            Regularization term for x latent factors

        y_reg : (float)
            Regularization term for y latent factors

        verbose : (bool)
            Whether or not to printout training progress
        """
        self.obs = obs
        self.n_factors = n_factors
        self.x_reg = x_reg
        self.y_reg = y_reg
        self.random_state = random_state
        self.verbose = verbose
        self.__set_model__()

    def __set_model__(self):
        """

        Paramas
        ======
        obs: shape (n x m)
        k: n_factors
        X: shape (n x k)
        Y: shape (m x k)

        obs_est = XY^T

        min (obs - obs_est)**2 + x_reg||X**2|| + y_reg||Y**2||
        :return:
        """
        self.X = np.random.normal((self.obs.shape[0], self.n_factors))
        self.Y = np.random.normal((self.obs.shape[1], self.n_factors))

    def als_step(self, type):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'x':
            # Precompute
            YTY = self.Y.T.dot(self.Y)
            reg_factor = np.eye(YTY.shape[0]) * self.x_reg

            for row_id in xrange(self.X.shape[0]):
                self.X[row_id, :] = np.linalg.solve((YTY + reg_factor), self.obs[row_id, :].dot(self.Y))

        elif type == 'y':
            # Precompute
            XTX = self.X.T.dot(self.X)
            reg_factor = np.eye(XTX.shape[0]) * self.y_reg

            for row_id in xrange(self.Y.shape[0]):
                self.Y[row_id, :] = np.linalg.solve((XTX + reg_factor), self.obs[:, row_id].T.dot(self.X))

    def train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self.verbose:
                print '\tcurrent iteration: {}'.format(ctr)
            self.als_step(type='x')
            self.als_step(type='y')
            ctr += 1

    def obs_est(self):
        """
        obs_est
        :return:
        """
        return np.matmul(self.X, self.Y.T)