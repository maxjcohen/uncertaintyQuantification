import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

from ucn.models import SMCN
from src.trainer import SMCNTrainer
from src.dataset import ARMADataset


def compare_stuff(model):
    print(
        "Sigma_x:\t",
        model._sigma_x.diag().detach().square().numpy().squeeze().round(2),
        f"\t({sigma_x})",
    )
    print(
        "Sigma_y:\t",
        model._sigma_y.diag().detach().square().numpy().squeeze().round(2),
        f"\t({sigma_y})",
    )
    print(
        "a:\t\t",
        model._g._linear.weight.detach().numpy().squeeze().round(2),
        f"\t({b})",
    )
    print(
        "b:\t\t", model._f._linear.weight.detach().numpy().squeeze().round(2), "\t(1)"
    )


# Define model
b = 0.75

sigma_x = 0.6
sigma_y = 0.2

n_obs = 10000

# Load model
arparams = np.array([b])
maparams = np.array([0])

arparams = np.r_[1, -arparams]
maparams = np.r_[1, maparams]
x_arma = arma_generate_sample(arparams, maparams, n_obs, scale=sigma_x)

y_arma = (
    x_arma
    + np.random.multivariate_normal(
        (0,), np.diag([sigma_y]), size=len(x_arma)
    ).squeeze()
)

# Plot ARMA
plt.figure(figsize=(25, 5))
plt.plot(y_arma[:250], label="observation")
plt.plot(x_arma[:250], lw=3, label="state")
plt.legend()
plt.title("ARMA model")


# Load model
d_emb = 1
d_out = 1
N = 50

model = SMCN(d_emb, d_out, n_particles=N)

# Load dataset
T = 25
batch_size = 8
epochs = 30

dataset = ARMADataset(y_arma, torch.zeros(len(y_arma), d_emb), T)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

# Print initial state
compare_stuff(model)

# Train
train_model = SMCNTrainer(model)
trainer = pl.Trainer(max_epochs=epochs, gpus=0)
trainer.fit(train_model, dataloader)

# Print final state
compare_stuff(model)
