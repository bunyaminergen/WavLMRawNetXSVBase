# Standard library imports
import math
from typing import Annotated

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """
    Sinc-based convolution layer for raw audio input.

    This layer performs convolution using parameterized Sinc functions
    as filters. It is inspired by the SincNet approach, where each filter
    is defined by two parameters: the low frequency and the band width.
    This allows for more interpretable filters and can lead to improved
    performance and faster convergence in some cases.

    Parameters
    ----------
    out_channels : int
        Number of filters (output channels).
    kernel_size : int
        Size of each filter in samples. Must be odd.
    sample_rate : int, optional
        Sampling rate of the audio signals. Defaults to 16000.
    in_channels : int, optional
        Number of input channels. Must be 1 for SincConv. Defaults to 1.
    stride : int, optional
        Convolution stride. Defaults to 1.
    padding : int, optional
        Convolution padding. Defaults to 0.
    min_low_hz : int, optional
        Minimum low cutoff frequency for the filters. Defaults to 50 Hz.
    min_band_hz : int, optional
        Minimum bandwidth for the filters. Defaults to 50 Hz.

    Attributes
    ----------
    out_channels : int
        Number of filters (output channels).
    kernel_size : int
        Size of each filter in samples.
    sample_rate : int
        Sampling rate used for the filters.
    stride : int
        Convolution stride.
    padding : int
        Convolution padding.
    min_low_hz : int
        Minimum low cutoff frequency for the filters.
    min_band_hz : int
        Minimum bandwidth for the filters.
    low_hz_ : nn.Parameter
        Trainable parameter for the low cutoff frequency of each filter.
    band_hz_ : nn.Parameter
        Trainable parameter for the bandwidth of each filter.

    Examples
    --------
    >>> import torch
    >>> sinc_conv_layer = SincConv(
    ...     out_channels=8,
    ...     kernel_size=31,
    ...     stride=2,
    ...     padding=3
    ... )
    >>> input_waveform = torch.randn(2, 1, 16000)
    >>> output = sinc_conv_layer(input_waveform)
    >>> output.shape  # doctest: +SKIP
    torch.Size([2, 8, 8000])
    """

    def __init__(
            self,
            out_channels: Annotated[int, "Number of filters (output channels)"],
            kernel_size: Annotated[int, "Filter size (must be odd)"],
            sample_rate: Annotated[int, "Sampling rate"] = 16000,
            in_channels: Annotated[int, "Number of input channels"] = 1,
            stride: Annotated[int, "Convolution stride"] = 1,
            padding: Annotated[int, "Convolution padding"] = 0,
            min_low_hz: Annotated[int, "Minimum low cutoff frequency"] = 50,
            min_band_hz: Annotated[int, "Minimum bandwidth"] = 50
    ) -> None:
        """
        Initialize the SincConv layer.

        Raises
        ------
        TypeError
            If any of the provided parameters is not an integer.
        ValueError
            If in_channels != 1, or if kernel_size is not odd, or if
            other parameters are out of acceptable ranges.
        """
        super(SincConv, self).__init__()

        for param_name, param_value in {
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "sample_rate": sample_rate,
            "in_channels": in_channels,
            "stride": stride,
            "padding": padding,
            "min_low_hz": min_low_hz,
            "min_band_hz": min_band_hz
        }.items():
            if not isinstance(param_value, int):
                raise TypeError(f"Expected '{param_name}' to be an integer.")

        if in_channels != 1:
            raise ValueError("SincConv only supports one input channel.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        band = (high_hz - low_hz) / self.out_channels

        self.low_hz_ = nn.Parameter(
            torch.linspace(low_hz, high_hz, self.out_channels)
        )
        self.band_hz_ = nn.Parameter(
            torch.full((self.out_channels,), band)
        )

        n_lin = torch.linspace(0, kernel_size - 1, kernel_size)
        window_ = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / (kernel_size - 1)
        )
        window_ = window_.float()
        self.register_buffer("window_", window_)

        half_kernel = (kernel_size - 1) // 2
        n_arr = torch.arange(-half_kernel, half_kernel + 1)
        n_ = 2 * math.pi * n_arr / self.sample_rate
        n_ = n_.float()
        self.register_buffer("n_", n_)

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input tensor of shape (B, 1, T)"]
    ) -> Annotated[torch.Tensor, "Output tensor of shape (B, out_channels, new_T)"]:
        """
        Forward pass of the SincConv layer.

        This computes the Sinc-based filters on-the-fly using the current
        values of `low_hz_` and `band_hz_`, and applies the filters via
        a 1D convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input waveform of shape (batch_size, 1, samples).

        Returns
        -------
        torch.Tensor
            Output features of shape
            (batch_size, out_channels, new_samples), depending on
            stride/padding choices.

        Examples
        --------
        >>> import torch
        >>> sinc_conv_layer = SincConv(out_channels=4, kernel_size=31)
        >>> x_test = torch.randn(2, 1, 32000)
        >>> y = sinc_conv_layer(x_test)
        >>> y.shape  # doctest: +SKIP
        torch.Size([2, 4, 32000])
        """
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2
        )
        band = (high - low)[:, None]

        f_low = low[:, None] * self.n_
        f_high = high[:, None] * self.n_

        band_pass_left = (2 * f_high).sin() / (1e-8 + 2 * f_high)
        band_pass_right = (2 * f_low).sin() / (1e-8 + 2 * f_low)
        band_pass = band_pass_left - band_pass_right

        band_pass = band_pass * self.window_

        band_pass = band_pass / (2 * band + 1e-8)

        filters = band_pass.unsqueeze(1).to(x.device)

        return F.conv1d(
            x,
            filters,
            stride=self.stride,
            padding=self.padding
        )


if __name__ == "__main__":
    sinc_conv_test_block = SincConv(
        out_channels=8,
        kernel_size=31,
        sample_rate=22050,
        stride=2,
        padding=3,
        min_low_hz=40,
        min_band_hz=30
    )

    test_input_sinc = torch.randn(2, 1, 16000)

    test_output_sinc = sinc_conv_test_block(test_input_sinc)
    print("SincConv Input :", test_input_sinc.shape)
    print("SincConv Output:", test_output_sinc.shape)
