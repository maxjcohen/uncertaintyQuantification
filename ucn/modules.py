import torch.nn as nn


class FFN(nn.Module):

    """Feed Forward Network Module."""

    def __init__(self, input_size, ouput_size, **kwargs):
        """TODO: to be defined.

        Parameters
        ----------
        input_size : TODO
        ouput_size : TODO


        """
        nn.Module.__init__(self, **kwargs)

        self._input_size = input_size
        self._ouput_size = ouput_size

        self._linear = nn.Linear(self._input_size, self._ouput_size, bias=False)
        self.activation = nn.Sigmoid()

        # Initialize with positive values only
        # TODO This is to be removed
        self._linear.weight.data = self._linear.weight.data.abs()

    def forward(self, x):
        return self._linear(x)
