import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import pytorch_lightning as pl


class SMCNTrainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, x):
        return self._model.forward(x)

    def training_step(self, batch, batch_idx):
        u, y = batch
        u = u.transpose(0, 1)
        y = y.transpose(0, 1)

        fisher = self._model.N > 1

        T = y.shape[0]
        if fisher:
            # Filter with observations
            y_hat = self._model(u, y=y, noise=True)

            # Smooth
            y_hat = self._model.smooth_pms(y_hat, self._model.I)
            particules = self._model.smooth_pms(self._model.particules, self._model.I)

            # Compute likelihood
            normal_y = MultivariateNormal(y.unsqueeze(-2), self._model._sigma_y)
            loss_y = - normal_y.log_prob(y_hat) * self._model.w

            normal_x = MultivariateNormal(particules[1:], self._model._sigma_x)
            loss_x = - normal_x.log_prob(particules[:T-1]) * self._model.w

            # Aggregate terms
            loss = loss_x.mean() + loss_y.mean()
        else:
            y_hat = self._model(u, noise=True)
            loss = F.mse_loss(y.squeeze(), y_hat.squeeze())

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
