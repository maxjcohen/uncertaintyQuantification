import torch
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

        self._linear = nn.Linear(self._input_size, self._ouput_size)
        self.activation = torch.tanh

    def forward(self, x):
        return self.activation(self._linear(x))


class ParticulesRNNCell(nn.RNNCell):

    """Docstring for RNN. """

    def __init__(self, input_size, hidden_size, **kwargs):
        """TODO: to be defined.

        Parameters
        ----------
        input_size : TODO
        hidden_size : TODO
        **kwargs : TODO


        """
        super().__init__(input_size=input_size, hidden_size=hidden_size, **kwargs)

        self._input_size = input_size
        self._hidden_size = hidden_size

    def forward(self, input_vector, hidden_vector):
        N = hidden_vector.shape[-2]
        input_vector = input_vector.reshape(-1, self._input_size).repeat(N, 1)

        return (
            super()
            .forward(input_vector, hidden_vector.view(-1, self._hidden_size))
            .view(hidden_vector.shape)
        )
