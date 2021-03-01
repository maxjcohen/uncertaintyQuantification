import torch
from torch.utils.data import Dataset


class ARMADataset(Dataset):
    def __init__(self, y, u, T):
        n_chunk = len(y) // T

        _y = torch.Tensor(y[: n_chunk * T])

        if len(_y.shape) == 1:
            _y = _y.unsqueeze(-1)

        self._y = _y.chunk(n_chunk)
        self._u = u.chunk(n_chunk)

    def __getitem__(self, idx):
        return (self._u[idx], self._y[idx])

    def __len__(self):
        return len(self._y)

class ARMAInitDataset(Dataset):
    def __init__(self, y, u):
        self._y = torch.Tensor(y).unsqueeze(-1)
        self._u = torch.Tensor(u).unsqueeze(-1)

    def __getitem__(self, idx):
        return (self._u[idx], self._y[idx])

    def __len__(self):
        return len(self._y)
