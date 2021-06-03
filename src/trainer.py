import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class SMCNTrainer(pl.LightningModule):
    def __init__(self, model, lr=None):
        super().__init__()
        self._model = model
        self.lr = lr or 2e-2
        self._hist = {
            "loss": []
        }
        for param_name in self._model.state_dict().keys():
            self._hist[param_name] = []
        self.sigma_x2 = torch.zeros(2, 2)

    def forward(self, x):
        return self._model.forward(x)

    def training_step(self, batch, batch_idx):
        u, y = batch
        u = u.transpose(0, 1)
        y = y.transpose(0, 1)

        fisher = self._model.N > 1

        for param_name, param in self._model.state_dict().items():
            self._hist[param_name].append(np.array(param.detach().cpu().squeeze()))
        if fisher:
            self._model(u=u, y=y, noise=True)
            # Update Sigma_x
            self.sigma_x2 = self._model.compute_sigma_x(u=u)*0.1 + self.sigma_x2*0.9
            self._model._sigma_x.data = self.sigma_x2
            return
        else:
            y_hat = self._model(u, noise=False)
            loss = F.mse_loss(y.squeeze(), y_hat.squeeze())

        # Log
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self._hist['loss'].append(loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
