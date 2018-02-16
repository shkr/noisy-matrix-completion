import torch
import torch.nn as nn
from torch.autograd import Variable


class MatrixDNN(torch.nn.Module):
    def __init__(self,
                 obs,
                 n_factors,
                 learning_rate,
                 l2_reg,
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

        l2_reg : (float)
            Regularization term for x latent factors

        verbose : (bool)
            Whether or not to printout training progress
        """
        super(MatrixDNN, self).__init__()

        self.obs = obs
        self.obs_tensor = Variable(torch.from_numpy(np.asarray(self.obs, dtype=np.float32)))
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
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
        self.X = nn.Embedding(num_embeddings=self.obs.shape[0],
                              embedding_dim=self.n_factors,
                              scale_grad_by_freq=True,
                              sparse=False)

        self.Y = nn.Embedding(num_embeddings=self.obs.shape[1],
                              embedding_dim=self.n_factors,
                              scale_grad_by_freq=True,
                              sparse=False)

        self.X_bias = torch.nn.Embedding(num_embeddings=self.obs.shape[0],
                                         embedding_dim=1,
                                         sparse=False)

        self.Y_bias = torch.nn.Embedding(self.obs.shape[1],
                                         embedding_dim=1,
                                         sparse=False)

        self.loss_func = nn.MSELoss(reduce=True)
        self.optim = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)

        self.sample_row, self.sample_col = self.obs.nonzero()
        self.n_samples = len(self.sample_row)
        self.training_indices = np.arange(self.n_samples)

    @staticmethod
    def const_as_var(*vars):
        out = []
        for v in vars:
            out.append(Variable(torch.LongTensor([v])))
        return out

    def forward(self, u, i):
        return torch.matmul(self.X(u), self.Y(i).t()) + self.X_bias(u) + self.Y_bias(i)

    def __call__(self, *args):
        return self.forward(*args)

    def sgd_minibatch(self):
        """
        SGD
        :param type:
        :return:
        """
        np.random.shuffle(self.training_indices)

        u = Variable(torch.LongTensor(self.sample_row[self.training_indices]))
        i = Variable(torch.LongTensor(self.sample_col[self.training_indices]))

        obs_est = torch.matmul(self.X.weight, self.Y.weight.t()) + self.X_bias.weight + self.Y_bias.weight.t()

        prediction = obs_est[self.sample_row[self.training_indices], self.sample_col[self.training_indices]]

        target = self.obs_tensor[self.sample_row[self.training_indices], self.sample_col[self.training_indices]]

        loss = self.loss_func(prediction, target)
        self.loss = loss
        total_loss = torch.Tensor([0])
        loss.backward()
        self.optim.step()
        return loss.data

    def sgd_step(self):
        """
        SGD
        :param type:
        :return:
        """
        np.random.shuffle(self.training_indices)
        total_loss = torch.Tensor([0])

        self.optim.zero_grad()

        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]

            [u, i] = self.const_as_var(u, i)

            prediction = self.forward(u, i)
            target = self.obs_tensor[u, i]

            loss = self.loss_func(prediction, target)
            loss.backward()
            self.optim.step()
            total_loss += loss.data
        return total_loss

    def train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        total_loss = torch.Tensor([0])
        while ctr <= n_iter:
            if ctr % 10 == 0 and self.verbose:
                print '\tcurrent iteration: {}'.format(ctr)
            total_loss += self.sgd_minibatch()
            ctr += 1
        print('Mean Loss :', (total_loss.numpy()[0] / n_iter))

    def obs_est(self):
        """
        obs_est
        :return:
        """
        res = torch.matmul(self.X.weight, self.Y.weight.t())
        res += self.X_bias.weight
        res += self.Y_bias.weight.t()
        return res.data.numpy()
