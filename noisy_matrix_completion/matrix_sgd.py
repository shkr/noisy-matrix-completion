import numpy as np


class MatrixSGD:
    def __init__(self,
                 obs,
                 n_factors,
                 learning_rate,
                 x_reg,
                 y_reg,
                 xb_reg,
                 yb_reg,
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
        self.learning_rate = learning_rate
        self.x_reg = x_reg
        self.y_reg = y_reg
        self.xb_reg = xb_reg
        self.yb_reg = yb_reg
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
        self.X = np.random.normal(size=(self.obs.shape[0], self.n_factors))
        self.Y = np.random.normal(size=(self.obs.shape[1], self.n_factors))

        self.X_bias = np.zeros(self.obs.shape[0])
        self.Y_bias = np.zeros(self.obs.shape[1])

        self.sample_row, self.sample_col = self.obs.nonzero()
        self.n_samples = len(self.sample_row)
        self.training_indices = np.arange(self.n_samples)

    def forward(self, u, i):
        return np.matmul(self.X[u, :], self.Y[i, :].T) + self.X_bias[u] + self.Y_bias[i]

    def sgd_step(self):
        """
        SGD
        :param type:
        :return:
        """
        np.random.shuffle(self.training_indices)

        for idx in self.training_indices:

            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.forward(u, i)
            e = (self.obs[u, i] - prediction)  # error

            # Update biases
            self.X_bias[u] += self.learning_rate * \
                                 (e - self.xb_reg * self.X_bias[u])
            self.Y_bias[i] += self.learning_rate * \
                                 (e - self.yb_reg * self.Y_bias[i])

            # Update embeddings
            self.X[u, :] += self.learning_rate * \
                                    (e * self.Y[i, :] - \
                                     self.x_reg * self.X[u, :])
            self.Y[i, :] += self.learning_rate * \
                                    (e * self.X[u, :] - \
                                     self.y_reg * self.Y[i, :])

    def train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self.verbose:
                print '\tcurrent iteration: {}'.format(ctr)
            self.sgd_step()
            ctr += 1

    def obs_est(self):
        """
        obs_est
        :return:
        """
        return np.matmul(self.X, self.Y.T) + self.X_bias[:, np.newaxis] + self.Y_bias[np.newaxis, :]
