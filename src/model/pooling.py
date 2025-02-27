# Standard library imports
from typing import Annotated

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatsPooling(nn.Module):
    """
    Attentive Statistics Pooling layer.

    This layer computes a weighted mean and standard deviation across
    the time dimension, where the weights (attention) are learned
    through a small neural network. The resulting output is a concatenation
    of the mean and standard deviation vectors, effectively producing
    a fixed-dimensional representation of variable-length sequences.

    Parameters
    ----------
    in_dim : int
        The number of input channels/features in each time frame.
    attention_dim : int, optional
        The internal dimension of the attention mechanism. Defaults to 128.

    Attributes
    ----------
    linear : nn.Linear
        A linear layer that projects the input features to the attention
        dimension (`attention_dim`).
    tanh : nn.Tanh
        Activation function applied after the linear layer.
    att_conv : nn.Conv1d
        A 1D convolution layer that produces attention weights across
        the time dimension.

    Examples
    --------
    >>> import torch
    >>> from src.model.pooling import AttentiveStatsPooling  # doctest: +SKIP
    >>> pooling_layer = AttentiveStatsPooling(in_dim=32, attention_dim=64)
    >>> x = torch.randn(2, 32, 100)
    >>> pooled_output = pooling_layer(x)
    >>> pooled_output.shape
    torch.Size([2, 64])

    References
    ----------
    * Okabe, Koji; Koshinaka, Takafumi; Shinoda, Koichi.
      "Attentive Statistics Pooling for Deep Speaker Embedding."
      In Proc. Interspeech 2018, 2018.

    * Pan, Lidong; He, Chunhao; Chang, Tieyuan.
      "External-Attentive Statistics Pooling for Text-Independent Speaker Verification."
      In Proc. IEEE 3rd Int. Conf. on Computer Communication and Artificial Intelligence (CCAI), 2023.

    * Zhang, Leying; Chen, Zhengyang; Qian, Yanmin.
      "Enroll-Aware Attentive Statistics Pooling for Target Speaker Verification."
      In Proc. Interspeech 2022, 2022.
    """

    def __init__(
            self,
            in_dim: Annotated[int, "Number of features in each time frame"],
            attention_dim: Annotated[int, "Attention dimension"] = 128
    ) -> None:
        """
        Initialize the AttentiveStatsPooling layer with the specified
        input dimension and attention dimension.

        Raises
        ------
        TypeError
            If `in_dim` or `attention_dim` is not an integer.
        ValueError
            If `in_dim` or `attention_dim` is less than or equal to 0.
        """
        super(AttentiveStatsPooling, self).__init__()

        if not isinstance(in_dim, int):
            raise TypeError("Expected 'in_dim' to be an integer.")
        if not isinstance(attention_dim, int):
            raise TypeError("Expected 'attention_dim' to be an integer.")
        if in_dim <= 0:
            raise ValueError("'in_dim' must be > 0.")
        if attention_dim <= 0:
            raise ValueError("'attention_dim' must be > 0.")

        self.linear = nn.Linear(in_dim, attention_dim)
        self.tanh = nn.Tanh()
        self.att_conv = nn.Conv1d(attention_dim, 1, kernel_size=1)

    def forward(
            self,
            x: Annotated[torch.Tensor, "(batch_size, features, time) input tensor"]
    ) -> Annotated[torch.Tensor, "(batch_size, features*2) output tensor"]:
        """
        Compute attentive mean and standard deviation across the time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, features, time).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, features*2), concatenating
            the mean and standard deviation after applying attention.

        Examples
        --------
        >>> import torch
        >>> pooling_layer = AttentiveStatsPooling(in_dim=32, attention_dim=64)
        >>> x_test = torch.randn(2, 32, 100)
        >>> output = pooling_layer(x)
        >>> output.shape
        torch.Size([2, 64])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected 'x' to be a torch.Tensor.")
        if x.dim() != 3:
            raise ValueError("Input tensor must have 3 dimensions (B, C, T).")

        out = self.linear(x.transpose(1, 2))
        out = self.tanh(out)

        out = out.transpose(1, 2)

        w = self.att_conv(out)

        w = F.softmax(w, dim=2)

        mean = torch.sum(x * w, dim=2)
        mean_sq = torch.sum((x ** 2) * w, dim=2)
        std = torch.sqrt(mean_sq - mean ** 2 + 1e-9)

        pooled = torch.cat([mean, std], dim=1)
        return pooled


if __name__ == "__main__":
    attentive_stats_pool_test_model = AttentiveStatsPooling(
        in_dim=32,
        attention_dim=64
    )

    x_test_attentive_pool = torch.randn(2, 32, 100)
    output_test_attentive_pool = attentive_stats_pool_test_model(x_test_attentive_pool)

    print("AttentiveStatsPooling Input Shape :", x_test_attentive_pool.shape)
    print("AttentiveStatsPooling Output Shape:", output_test_attentive_pool.shape)
