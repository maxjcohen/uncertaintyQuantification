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
        nn.RNNCell.__init__(
            self, input_size=input_size, hidden_size=hidden_size, **kwargs
        )

        self._input_size = input_size
        self._hidden_size = hidden_size

    def forward(self, input_vector, hidden_vector):
        input_vector = (
            input_vector.unsqueeze(-2)
            .expand(hidden_vector.shape)
            .reshape(-1, self._input_size)
        )

        return (
            super()
            .forward(input_vector, hidden_vector.view(-1, self._input_size))
            .view(hidden_vector.shape)
        )
