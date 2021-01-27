import math

import torch
import torch.nn as nn
import numpy as np

from ucn.models import SMCN

# Shapes
T = 5
d_emb = 4
d_out = 2

N = 3

bs = 2

# Inputs and observations
u_tild = torch.randn(T, bs, d_emb)
u_tild  # (T, bs, d_emb)

y = torch.randn(T, bs, d_out)
y  # (T, bs, d_out)

model = SMCN(d_emb, d_out, n_particles=N)

model(u_tild, noise=True)
