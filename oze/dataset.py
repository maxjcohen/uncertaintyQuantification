import numpy as np
from torch.utils.data import Dataset


class OzeDataset(Dataset):
    def __init__(self, y, u, T):
        self._T = T
        self._normalization_const = {}

        y = self.normalize(y, label="observation")
        u = self.normalize(u, label="command")

        self._y = torch.Tensor(y)
        self._u = torch.Tensor(u)

        if len(self._y.shape) == 1:
            self._y = self._y.unsqueeze(-1)

    def normalize(self, array, label=None):
        array_min = array.min(axis=0, keepdims=True)
        array_max = array.max(axis=0, keepdims=True)

        if label:
            self._normalization_const[label] = (array_min, array_max)

        return (array - array_min) / (array_max - array_min + np.finfo(np.float32).eps)

    def rescale(self, array, label):
        try:
            array_min, array_max = self._normalization_const[label]
        except KeyError:
            raise NameError(f"Can't rescale array with unknown label {label}.")

        return array * (array_max - array_min + np.finfo(np.float32).eps) + array_min

    @property
    def y(self):
        return self.rescale(self._y, label="observation")

    def __getitem__(self, idx):
        return (self._u[idx : idx + self._T], self._y[idx : idx + self._T])

    def __len__(self):
        return len(self._y) - self._T
