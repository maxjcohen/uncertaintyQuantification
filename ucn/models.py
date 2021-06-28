import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from .modules import FFN, ParticulesRNNCell


class SMCN(nn.Module):

    """Sequential Monte Carlo Network."""

    def __init__(self, input_size, latent_size, output_size, n_particles=1):
        """TODO: to be defined.

        Parameters
        ----------
        input_size : TODO
        output_size : TODO
        n_particles : TODO, optional


        """
        nn.Module.__init__(self)

        self._input_size = input_size
        self._latent_size = latent_size
        self._output_size = output_size
        self._n_particles = n_particles

        self._f = FFN(self._latent_size, self._output_size)
        self._g = ParticulesRNNCell(self._input_size, self._latent_size)
        self._input_model = nn.GRU(self._input_size, self._input_size, num_layers=3)

        self._sigma_x = nn.Parameter(
            torch.rand(self._latent_size), requires_grad=True
        )
        self._sigma_y = nn.Parameter(
            torch.diag(torch.abs(torch.rand(self._output_size))), requires_grad=True
        )

        self.softmax = nn.Softmax(dim=1)

    @property
    def sigma_x2(self):
        return torch.diag(self._sigma_x)

    @sigma_x2.setter
    def sigma_x2(self, matrix):
        self._sigma_x.data = torch.diag(matrix)

    @property
    def sigma_y2(self):
        return self._sigma_y

    def forward(self, u, y=None, noise=False):
        # N should be 1 if there is no noise
        assert noise or self.N == 1
        fisher = y is not None

        T = u.shape[0]
        bs = u.shape[1]

        predictions = []
        self._particules = []
        self._I = []

        # Generate initial particles
        x = torch.zeros(bs, self.N, self._latent_size, device=u.device)
        self._eta = MultivariateNormal(
            loc=torch.zeros(x.shape), covariance_matrix=self.sigma_x2
        )

        self._particules.append(x)

        # Compute weights
        self._W = []
        y_hat = self._f(x)
        predictions.append(y_hat)

        # Iterate k through time
        for k in range(1, T):
            if fisher:
                self.w = self.compute_weights(y[k-1], predictions[k-1].transpose(0, 1))
                self._W.append(self.w)

                # Select sampled indices from previous time step
                I = torch.multinomial(self.w, self.N, replacement=True)
                self._I.append(I)
                x = self.__class__.select_indices(x, I)

            # Compute new hidden state
            x = self._g(u[k], x)
            if noise:
                x = x + self._eta.sample()
            self._particules.append(x)

            # Compute new weights
            y_hat = self._f(x)
            predictions.append(y_hat)
        if fisher:
            self.w = self.compute_weights(y[-1], predictions[-1].transpose(0, 1))
            self._W.append(self.w)
        return torch.stack(predictions)

    def compute_weights(self, y, y_hat):
        _normal_y = MultivariateNormal(
            y, covariance_matrix=self.sigma_y2
        )
        w = _normal_y.log_prob(y_hat).T
        w = self.softmax(w)
        return w

    @staticmethod
    def select_indices(x, I):
        return torch.cat(
            [x[batch_idx, particule_idx] for batch_idx, particule_idx in enumerate(I)]
        ).view(x.shape)

    # @classmethod
    def smooth_pms(self, x, I):
        N = I.shape[-1]
        T = x.shape[0]
        bs = x.shape[1]

        # Initialize flat indexing
        I_flat = torch.zeros((T, bs, N), dtype=torch.long)
        I_flat[-1] = torch.arange(N)

        # Fill flat indexing with reversed indexing
        for k in reversed(range(T - 1)):
            I_flat[k] = self.__class__.select_indices(I[k], I_flat[k + 1])

        self._I_flat = I_flat
        # Stack all selected particles
        return torch.stack(
            [self.__class__.select_indices(x_i, I_i) for x_i, I_i in zip(x, I_flat)]
        )

    def compute_cost(self, u, y):
        # Smooth
        particules = self.smooth_pms(self.particules, self.I).detach()

        # Compute likelihood
        normal_y = MultivariateNormal(y.unsqueeze(-2), covariance_matrix=self.sigma_y2)
        loss_y = -normal_y.log_prob(self._f(particules)) * self.w.detach()
        loss_y = loss_y.sum(-1)

        normal_x = MultivariateNormal(particules[1:], covariance_matrix=self.sigma_x2)
        loss_x = -normal_x.log_prob(self._g(u[:-1], particules[:-1])) * self.w.detach()
        loss_x = loss_x.sum(-1)

        # Aggregate terms
        return loss_x.mean() + loss_y.mean()

    def compute_sigma_y(self, u, y):
        # Smooth
        particules = self.smooth_pms(self.particules, self.I).detach()

        sigma_y2 = y.unsqueeze(-1) - self._f(particules)

        # Compute square
        sigma_y2 = sigma_y2.square()

        # Sum on time steps
        sigma_y2 = sigma_y2.mean(0)

        # Sum on particules
        sigma_y2 = sigma_y2 * self.w.unsqueeze(-1).detach()
        sigma_y2 = sigma_y2.sum(axis=1)

        # Average on batches
        sigma_y2 = sigma_y2.mean(0)

        return sigma_y2

    def compute_sigma_x(self, u):
        # Smooth
        particules = self.smooth_pms(self.particules, self.I).detach()

        sigma_x2 = particules[1:] - self._g(u[:-1], particules[:-1])

        # Compute square
        sigma_x2 = sigma_x2.square()

        # Sum on time steps
        sigma_x2 = sigma_x2.mean(0)

        # Sum on particules
        sigma_x2 = sigma_x2 * self.w.unsqueeze(-1).detach()
        sigma_x2 = sigma_x2.sum(axis=1)

        # Average on batches
        sigma_x2 = sigma_x2.mean(0)
        return torch.diag(sigma_x2)

    @property
    def N(self):
        return self._n_particles

    @N.setter
    def N(self, n):
        self._n_particles = n

    @property
    def I(self):
        return torch.stack(self._I)

    @property
    def W(self):
        return torch.stack(self._W)

    @property
    def particules(self):
        return torch.stack(self._particules)
