# Standard library imports
from typing import Annotated

# Third-party imports
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.

    This block implements the squeeze-and-excitation mechanism that
    adaptively recalibrates channel-wise feature responses by explicitly
    modelling inter-dependencies between channels.

    Parameters
    ----------
    channel : int
        Number of input/output channels.
    reduction : int, optional
        Reduction ratio in the fully connected layers. Defaults to 8.

    Examples
    --------
    >>> se_block_ = SEBlock(channel=16)
    >>> x = torch.randn(4, 16, 32)
    >>> y = se_block_(x)
    >>> y.shape
    torch.Size([4, 16, 32])

    References
    ----------
    * Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-Excitation Networks."
      2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018).
    """

    def __init__(
            self,
            channel: Annotated[int, "Number of input/output channels"],
            reduction: Annotated[int, "Reduction ratio"] = 8
    ) -> None:
        """
        Initialize the SEBlock.

        Parameters
        ----------
        channel : int
            Number of input/output channels.
        reduction : int, optional
            Reduction ratio. Defaults to 8.

        Raises
        ------
        TypeError
            If channel or reduction is not an integer.
        ValueError
            If channel <= 0 or reduction <= 0.
        """
        super(SEBlock, self).__init__()

        if not isinstance(channel, int):
            raise TypeError("Expected 'channel' to be an integer.")
        if not isinstance(reduction, int):
            raise TypeError("Expected 'reduction' to be an integer.")
        if channel <= 0:
            raise ValueError("'channel' must be > 0.")
        if reduction <= 0:
            raise ValueError("'reduction' must be > 0.")

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input tensor of shape (B, C, T)"]
    ) -> Annotated[torch.Tensor, "Output tensor of shape (B, C, T)"]:
        """
        Forward pass of the SEBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (batch_size, channels, length).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (batch_size, channels, length),
            recalibrated by channel attention.

        Examples
        --------
        >>> se_block = SEBlock(channel=16)
        >>> x_ = torch.randn(4, 16, 32)
        >>> y_ = se_block(x_)
        >>> y_.shape
        torch.Size([4, 16, 32])
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    A single residual block for 1D convolutional networks with optional
    Squeeze-and-Excitation (SE) block.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int, optional
        Convolution kernel size. Defaults to 3.
    dilation : int, optional
        Dilation rate of the convolution. Defaults to 1.
    use_se : bool, optional
        Whether to use Squeeze-and-Excitation block. Defaults to True.

    Examples
    --------
    >>> import torch
    >>> from src.model.block import ResidualBlock  # doctest: +SKIP
    >>> block = ResidualBlock(channels=17, kernel_size=5, dilation=2, use_se=False)
    >>> x = torch.randn(4, 17, 32)
    >>> y = block(x)
    >>> y.shape
    torch.Size([4, 17, 32])
    """

    def __init__(
            self,
            channels: Annotated[int, "Number of input/output channels"],
            kernel_size: Annotated[int, "Convolution kernel size"] = 3,
            dilation: Annotated[int, "Dilation rate"] = 1,
            use_se: Annotated[bool, "Use SE block"] = True
    ) -> None:
        """
        Initialize the ResidualBlock.

        Parameters
        ----------
        channels : int
            Number of input/output channels.
        kernel_size : int, optional
            Convolution kernel size. Defaults to 3.
        dilation : int, optional
            Dilation rate of the convolution. Defaults to 1.
        use_se : bool, optional
            Whether to include a squeeze-and-excitation block.
            Defaults to True.

        Raises
        ------
        TypeError
            If channels, kernel_size, or dilation is not an integer
            or use_se is not a bool.
        ValueError
            If channels <= 0, kernel_size <= 0, or dilation <= 0.
        """
        super(ResidualBlock, self).__init__()

        if not isinstance(channels, int):
            raise TypeError("Expected 'channels' to be an integer.")
        if not isinstance(kernel_size, int):
            raise TypeError("Expected 'kernel_size' to be an integer.")
        if not isinstance(dilation, int):
            raise TypeError("Expected 'dilation' to be an integer.")
        if not isinstance(use_se, bool):
            raise TypeError("Expected 'use_se' to be a bool.")
        if channels <= 0:
            raise ValueError("'channels' must be > 0.")
        if kernel_size <= 0:
            raise ValueError("'kernel_size' must be > 0.")
        if dilation <= 0:
            raise ValueError("'dilation' must be > 0.")

        self.use_se = use_se
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.norm = nn.InstanceNorm1d(channels, affine=True)

        if self.use_se:
            self.se = SEBlock(channel=channels)
        else:
            self.se = None

        self.act = nn.Mish()

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input tensor of shape (B, C, T)"]
    ) -> Annotated[torch.Tensor, "Output tensor of shape (B, C, T)"]:
        """
        Forward pass of the ResidualBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (batch_size, channels, length).

        Returns
        -------
        torch.Tensor
            Output feature map of the same shape as input.

        Examples
        --------
        >>> block = ResidualBlock(channels=17, kernel_size=5, dilation=2, use_se=False)
        >>> x_test = torch.randn(4, 17, 32)
        >>> y = block(x_test)
        >>> y.shape
        torch.Size([4, 17, 32])
        """
        identity = x
        out = self.conv(x)
        out = self.norm(out)

        if self.se is not None:
            out = self.se(out)

        out = self.act(out)
        return identity + out


class ResidualStack(nn.Module):
    """
    A stack of ResidualBlock modules.

    Parameters
    ----------
    channels : int
        Number of input/output channels for each block.
    kernel_size : int
        Convolution kernel size for each block.
    dilation : int
        Dilation rate for the convolutions.
    use_se : bool
        Whether to include an SEBlock in each residual block.
    num_blocks : int
        Number of residual blocks in the stack.

    Examples
    --------
    >>> import torch
    >>> from src.model.block import ResidualStack  # doctest: +SKIP
    >>>  stack = ResidualStack(channels=17, kernel_size=3, dilation=1, use_se=True, num_blocks=4)
    >>> x = torch.randn(4, 17, 32)
    >>> y = stack(x)
    >>> y.shape
    torch.Size([4, 17, 32])
    """

    def __init__(
            self,
            channels: Annotated[int, "Number of input/output channels"],
            kernel_size: Annotated[int, "Convolution kernel size"],
            dilation: Annotated[int, "Dilation rate"],
            use_se: Annotated[bool, "Use Squeeze-and-Excitation"],
            num_blocks: Annotated[int, "Number of residual blocks"]
    ) -> None:
        """
        Initialize the ResidualStack.

        Parameters
        ----------
        channels : int
            Number of input/output channels for each block.
        kernel_size : int
            Convolution kernel size for each block.
        dilation : int
            Dilation rate for the convolutions.
        use_se : bool
            Whether to include an SEBlock in each residual block.
        num_blocks : int
            Number of residual blocks in the stack.

        Raises
        ------
        TypeError
            If any of the input parameters have incorrect types.
        ValueError
            If any of the integer parameters are <= 0.
        """
        super().__init__()

        if not isinstance(channels, int):
            raise TypeError("Expected 'channels' to be an integer.")
        if not isinstance(kernel_size, int):
            raise TypeError("Expected 'kernel_size' to be an integer.")
        if not isinstance(dilation, int):
            raise TypeError("Expected 'dilation' to be an integer.")
        if not isinstance(use_se, bool):
            raise TypeError("Expected 'use_se' to be a bool.")
        if not isinstance(num_blocks, int):
            raise TypeError("Expected 'num_blocks' to be an integer.")

        if channels <= 0:
            raise ValueError("'channels' must be > 0.")
        if kernel_size <= 0:
            raise ValueError("'kernel_size' must be > 0.")
        if dilation <= 0:
            raise ValueError("'dilation' must be > 0.")
        if num_blocks <= 0:
            raise ValueError("'num_blocks' must be > 0.")

        self.blocks = nn.ModuleList([
            ResidualBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                use_se=use_se
            )
            for _ in range(num_blocks)
        ])

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input tensor of shape (B, C, T)"]
    ) -> Annotated[torch.Tensor, "Output tensor of shape (B, C, T)"]:
        """
        Forward pass for the ResidualStack.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (batch_size, channels, length).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (batch_size, channels, length)
            after passing through all residual blocks.

        Examples
        --------
        >>> stack = ResidualStack(channels=17, kernel_size=3, dilation=1, use_se=True, num_blocks=4)
        >>> test_x = torch.randn(4, 17, 32)
        >>> y = stack(test_x)
        >>> y.shape
        torch.Size([4, 17, 32])
        """
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    se_block_test = SEBlock(channel=16, reduction=9)
    test_input_se = torch.randn(4, 16, 32)
    se_block_output = se_block_test(test_input_se)
    print("SEBlock Output:", se_block_output.shape)

    residual_block_test = ResidualBlock(
        channels=17,
        kernel_size=5,
        dilation=2,
        use_se=False
    )
    test_input_res = torch.randn(4, 17, 32)
    residual_block_output = residual_block_test(test_input_res)
    print("ResidualBlock Output:", residual_block_output.shape)

    stack_test = ResidualStack(
        channels=17,
        kernel_size=3,
        dilation=1,
        use_se=True,
        num_blocks=4
    )
    test_input_stack = torch.randn(4, 17, 32)
    stack_output = stack_test(test_input_stack)
    print("ResidualStack Output:", stack_output.shape)
