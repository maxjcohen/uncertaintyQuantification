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
        self._g = nn.RNNCell(self._input_size, self._input_size, nonlinearity='tanh')

        self._sigma_x = nn.Parameter(
            torch.diag(torch.abs(torch.rand(self._input_size))), requires_grad=True
        )
        self._sigma_y = nn.Parameter(
            torch.diag(torch.abs(torch.rand(self._output_size))), requires_grad=True
        )

        # Load noise distribution
        self._eta = MultivariateNormal(
            torch.zeros(self.N, self._input_size), self._sigma_x
        )
        self._eps = MultivariateNormal(
            torch.zeros(self.N, self._output_size), self._sigma_y
        )

        # Load pdf around observation y
        self._normal_y = MultivariateNormal(torch.zeros(1), self._sigma_y)

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

        # Iterate k through time
        for k, u_k in enumerate(u):
            # Compute hidden state
            u_k = u_k.repeat_interleave(self.N, dim=0)
            x = x.view(-1, self._input_size)
            x = self._g(u_k, x).view(-1, self.N, self._input_size)

            if noise:
                x = x + self._eta.sample((bs,))
            if fisher:
                self._particules.append(x)
                x = x.detach()

            # Compute predictions
            y_hat = self._f(x)
            if noise:
                y_hat = y_hat + self._eps.sample((bs,))
            predictions.append(y_hat)

            if fisher:
                y_hat = y_hat.detach()

                # Compute sampling weights
                self._normal_y.loc = y[k]
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
        return torch.stack(
            [cls.resample(x_i, I_i) for x_i, I_i in zip(x, I_flat)]
        )

    @property
    def N(self):
        return self._n_particles

    @N.setter
    def N(self, n):
        self._n_particles = n

        # Load noise distribution
        self._eta = MultivariateNormal(
            torch.zeros(self.N, self._input_size), self._sigma_x
        )

    @property
    def I(self):
        return torch.stack(self._I[:-1])

    @property
    def particules(self):
        return torch.stack(self._particules)
