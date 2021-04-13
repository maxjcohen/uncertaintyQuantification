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
            torch.log(torch.diag(torch.rand(self._latent_size))), requires_grad=True
        )
        self._sigma_y = nn.Parameter(
            torch.log(torch.diag(torch.rand(self._output_size))), requires_grad=True
        )

        self.softmax = nn.Softmax(dim=1)

    @property
    def sigma_x(self):
        return torch.exp(self._sigma_x)

    @property
    def sigma_y(self):
        return torch.exp(self._sigma_y)

    def forward(self, u, y=None, noise=False):
        # N should be 1 if there is no noise
        assert noise or self.N == 1
        fisher = y is not None

        T = u.shape[0]
        bs = u.shape[1]

        predictions = []
        self._particules = []
        self._I = []

        u = self._input_model(u)[0]

        x = x + torch.randn(size=x.shape) * self.sigma_x.sqrt()
        self._particules.append(x)

        # Compute weights
        self._W = []
        y_hat = self._f(x)
        predictions.append(y_hat)

        # Iterate k through time
        for k in range(1, T):
            if fisher:
                self._normal_y = MultivariateNormal(
                    y[k - 1], covariance_matrix=self.sigma_y
                )
                self.w = self._normal_y.log_prob(y_hat.transpose(0, 1)).T
                self.w = self.softmax(self.w)
                self._W.append(self.w)

                # Resample previous time step
                I = torch.multinomial(self.w, self.N, replacement=True)
                self._I.append(I)
                x = self.__class__.resample(x, I)

            # Compute new hidden state
            x = self._g(u[k], x)
            if noise:
                x = x + self._eta.sample()
                self._particules.append(torch.Tensor(x))

            # Compute new weights
            y_hat = self._f(x)
            predictions.append(y_hat)
        if fisher:
            self._normal_y = MultivariateNormal(y[-1], covariance_matrix=self.sigma_y)
            self.w = self._normal_y.log_prob(y_hat.transpose(0, 1)).T
            self.w = self.softmax(self.w)
            self._W.append(self.w)
        return torch.stack(predictions)

    @staticmethod
    def resample(x, I):
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
            I_flat[k] = self.__class__.resample(I[k], I_flat[k + 1])

        self._I_flat = I_flat
        # Stack all selected particles
        return torch.stack(
            [self.__class__.resample(x_i, I_i) for x_i, I_i in zip(x, I_flat)]
        )

    def compute_cost(self, u, y):
        # Smooth
        particules = self.smooth_pms(self.particules, self.I).detach()

        # Compute likelihood
        normal_y = MultivariateNormal(y.unsqueeze(-2), covariance_matrix=self.sigma_y)
        loss_y = -normal_y.log_prob(self._f(particules)) * self.w.detach()
        loss_y = loss_y.sum(-1)

        normal_x = MultivariateNormal(particules[1:], covariance_matrix=self.sigma_x)
        loss_x = -normal_x.log_prob(self._g(u[:-1], particules[:-1])) * self.w.detach()
        loss_x = loss_x.sum(-1)

        # Aggregate terms
        return loss_x.mean() + loss_y.mean()

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
