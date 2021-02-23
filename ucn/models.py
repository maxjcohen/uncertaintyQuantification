import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from .modules import FFN


class SMCN(nn.Module):

    """Sequential Monte Carlo Network."""

    def __init__(self, input_size, output_size, n_particles=1):
        """TODO: to be defined.

        Parameters
        ----------
        input_size : TODO
        output_size : TODO
        n_particles : TODO, optional


        """
        nn.Module.__init__(self)

        self._input_size = input_size
        self._output_size = output_size
        self._n_particles = n_particles

        self._f = FFN(self._input_size, self._output_size)
        self._g = FFN(self._input_size, self._input_size)

        self._sigma_x = nn.Parameter(
            torch.cholesky(torch.diag(torch.rand(self._input_size))), requires_grad=True
        )
        self._sigma_y = nn.Parameter(
            torch.cholesky(torch.diag(torch.rand(self._output_size))),
            requires_grad=True,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, u, y=None, noise=False):
        # N should be 1 if there is no noise
        assert noise or self.N == 1
        fisher = y is not None

        T = u.shape[0]
        bs = u.shape[1]

        predictions = []
        self._particules = []
        self._I = [torch.arange(self.N, device=u.device).expand(bs, self.N)]

        # Initial hidden state
        x = torch.randn(bs, self.N, self._input_size, device=u.device)

        self._eta = MultivariateNormal(torch.zeros(x.shape), scale_tril=self._sigma_x)
        # Iterate k through time
        for k, u_k in enumerate(u):
            # Compute hidden state
            x = self._g(x)

            if noise:
                x = x + self._eta.sample()
            self._particules.append(x)

            # Compute predictions
            y_hat = self._f(x)
            predictions.append(y_hat)

            if fisher:
                # Compute sampling weights
                self._normal_y = MultivariateNormal(y[k], scale_tril=self._sigma_y)
                self.w = self._normal_y.log_prob(y_hat.transpose(0, 1)).T.detach()
                self.w = self.softmax(self.w)
                I = torch.multinomial(self.w, self.N, replacement=True)
                x = self.__class__.resample(x, I)

                self._I.append(I)

        return torch.stack(predictions)

    @staticmethod
    def resample(x, I):
        return torch.cat(
            [x[batch_idx, particule_idx] for batch_idx, particule_idx in enumerate(I)]
        ).view(x.shape)

    @classmethod
    def smooth_pms(cls, x, I):
        T, N = I.shape[0], I.shape[-1]

        # Initialize flat indexing
        I_flat = torch.zeros(I.shape, dtype=torch.long)
        I_flat[-1] = torch.arange(N)

        # Fill flat indexing with reversed indexing
        for k in reversed(range(T - 1)):
            I_flat[k] = cls.resample(I[k + 1], I_flat[k + 1])

        # Stack all selected particles
        return torch.stack([cls.resample(x_i, I_i) for x_i, I_i in zip(x, I_flat)])

    def compute_cost(self, u, y):
        # Filter with observations
        self(u, y=y, noise=True)

        # Smooth
        particules = self.smooth_pms(self.particules, self.I).detach()

        # Compute likelihood
        normal_y = MultivariateNormal(y.unsqueeze(-2), scale_tril=self._sigma_y)
        loss_y = -normal_y.log_prob(self._f(particules)) * self.w.detach()

        normal_x = MultivariateNormal(particules[1:], scale_tril=self._sigma_x)
        loss_x = -normal_x.log_prob(self._g(particules[:-1])) * self.w.detach()

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
        return torch.stack(self._I[:-1])

    @property
    def particules(self):
        return torch.stack(self._particules)
