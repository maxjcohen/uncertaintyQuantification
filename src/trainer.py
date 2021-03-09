import torch
import torch.nn.functional as F
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

        if fisher:
            self._model(u=u, y=y, noise=True)
            loss = self._model.compute_cost(u=u, y=y)
        else:
            y_hat = self._model(u, noise=True)
            loss = F.mse_loss(y.squeeze(), y_hat.squeeze())

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
