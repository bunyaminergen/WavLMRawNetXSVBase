# Standard library imports
from typing import Annotated

# Third-party imports
import torch
import torch.nn as nn
from transformers import WavLMModel


class RawNetX(nn.Module):
    """
    RawNetX model for speech feature extraction.

    This model uses a Sinc-based convolution, residual blocks,
    and pooling to produce embeddings of a specified dimension.
    It is designed for raw audio input (waveform).

    Parameters
    ----------
    sinc_conv : nn.Module
        A convolution-like layer (e.g., SincConv or standard Conv1d)
        for initial feature extraction from the raw waveform.
    residual_blocks : nn.Module
        A container (e.g., nn.Sequential) of residual blocks for
        further feature extraction.
    pooling_layer : nn.Module
        A pooling layer that reduces the time dimension to a fixed
        representation (e.g., mean+std pooling).
    sinc_out_channels : int
        Number of output channels from the `sinc_conv`.
    rawnetx_embedding_dim : int
        Output embedding dimension for the final layer.

    Attributes
    ----------
    sinc_conv : nn.Module
        The initial convolution layer.
    first_norm : nn.InstanceNorm1d
        Instance normalization layer for the convolution output.
    first_act : nn.Module
        Activation function applied after the normalization layer.
    blocks : nn.Module
        Container of residual blocks for feature extraction.
    pooling_layer : nn.Module
        Aggregation layer producing a global representation.
    emb_fc : nn.Linear
        Fully connected layer mapping the pooled representation
        to the embedding dimension.
    emb_act : nn.Module
        Activation function after the embedding FC layer.

    Examples
    --------
    >>> from src.model.block import ResidualStack
    >>> sinc_conv = nn.Conv1d(1, 128, kernel_size=251, stride=2, padding=125)
    >>> residual_stack_test = ResidualStack(128, 251, 1, True, 4)
    >>>
    >>> class MeanStdPoolingTest(nn.Module):
    ...     @staticmethod
    ...     def forward(test_x):
    ...         mean = test_x.mean(dim=2)
    ...         std =test_x.std(dim=2)
    ...         out = torch.cat([mean, std], dim=1)
    ...         return out.unsqueeze(-1)
    >>>
    >>> pooling_layer = MeanStdPoolingTest()
    >>>
    >>> rawnetx_model = RawNetX(
    ...     sinc_conv=sinc_conv,
    ...     residual_blocks=residual_stack,
    ...     pooling_layer=pooling_layer,
    ...     sinc_out_channels=128,
    ...     rawnetx_embedding_dim=128
    ... )
    >>> x = torch.randn(2, 1, 16000)
    >>> emb = rawnetx_model(x)
    >>> emb.shape
    torch.Size([2, 128])
    """

    def __init__(
            self,
            sinc_conv: Annotated[nn.Module, "Initial convolution layer"],
            residual_blocks: Annotated[nn.Module, "Sequential container of blocks"],
            pooling_layer: Annotated[nn.Module, "Pooling layer"],
            sinc_out_channels: Annotated[int, "Output channels of the sinc_conv"],
            rawnetx_embedding_dim: Annotated[int, "Final embedding dimension"]
    ) -> None:
        """
        Initialize the RawNetX model with specified components.
        """
        super().__init__()

        if not isinstance(sinc_conv, nn.Module):
            raise TypeError("Expected 'sinc_conv' to be an nn.Module.")
        if not isinstance(residual_blocks, nn.Module):
            raise TypeError(
                "Expected 'residual_blocks' to be an nn.Module."
            )
        if not isinstance(pooling_layer, nn.Module):
            raise TypeError("Expected 'pooling_layer' to be an nn.Module.")
        if not isinstance(sinc_out_channels, int):
            raise TypeError(
                "Expected 'sinc_out_channels' to be an integer."
            )
        if not isinstance(rawnetx_embedding_dim, int):
            raise TypeError(
                "Expected 'rawnetx_embedding_dim' to be an integer."
            )

        self.sinc_conv = sinc_conv
        self.first_norm = nn.InstanceNorm1d(sinc_out_channels, affine=True)
        self.first_act = nn.Mish()

        self.blocks = residual_blocks
        self.pooling_layer = pooling_layer

        self.emb_fc = nn.Linear(
            sinc_out_channels * 2, rawnetx_embedding_dim
        )
        self.emb_act = nn.ReLU()

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input waveform of shape (B, C, T)"]
    ) -> Annotated[torch.Tensor, "Output embeddings of shape (B, D)"]:
        """
        Forward pass through the RawNetX model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, samples).

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (batch_size, embedding_dim).

        Examples
        --------
        >>> import torch
        >>> rawnetx_model = RawNetX(... )  # doctest: +SKIP
        >>> x_test = torch.randn(2, 1, 16000)
        >>> emb_test = rawnetx_model(x_test)  # doctest: +SKIP
        >>> emb.shape  # doctest: +SKIP
        torch.Size([2, 128])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected 'x' to be a torch.Tensor.")

        x = self.sinc_conv(x)
        x = self.first_norm(x)
        x = self.first_act(x)

        x = self.blocks(x)

        x = self.pooling_layer(x)

        emb = self.emb_fc(x.squeeze(-1))
        emb = self.emb_act(emb)
        return emb


class WavLMLarge(nn.Module):
    """
    Wrapper class for the WavLM Large model from Hugging Face Transformers.

    This model takes raw waveforms as input (shape: (batch_size, samples))
    and returns an embedding by taking the average over the last hidden
    state. It is typically used for tasks such as speech recognition or
    speaker embedding.

    Parameters
    ----------
    wavlm_path : str, optional
        Path or identifier for the pretrained WavLM model from Hugging Face.
        Defaults to "microsoft/wavlm-large".

    Attributes
    ----------
    wavlm : WavLMModel
        The pretrained WavLM Large model.

    Examples
    --------
    >>> import torch
    >>> wavlm_model = WavLMLarge()
    >>> waveforms = torch.randn(2, 16000)
    >>> emb = wavlm_model(waveforms)
    >>> emb.shape
    torch.Size([2, 1024])
    """

    def __init__(
            self,
            wavlm_path: Annotated[
                str, "Path/identifier for pretrained WavLM model"
            ] = "microsoft/wavlm-large"
    ) -> None:
        """
        Initialize WavLMLarge with the specified or default model path.
        """
        super().__init__()
        if not isinstance(wavlm_path, str):
            raise TypeError("Expected 'wavlm_path' to be a string.")

        self.wavlm = WavLMModel.from_pretrained(wavlm_path)

    def forward(
            self,
            waveforms: Annotated[torch.Tensor, "Raw waveforms (B, T)"]
    ) -> Annotated[torch.Tensor, "WavLM embeddings of shape (B, D)"]:
        """
        Forward pass through the WavLM model.

        Parameters
        ----------
        waveforms : torch.Tensor
            Input tensor of shape (batch_size, samples).

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (batch_size, hidden_dim), where
            hidden_dim corresponds to the WavLM model's output dimension.

        Examples
        --------
        >>> import torch
        >>> wavlm_model = WavLMLarge()
        >>> waveforms_test = torch.randn(2, 16000)
        >>> emb_test = wavlm_model(waveforms_test)
        >>> emb_test.shape
        torch.Size([2, 1024])
        """
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("Expected 'waveforms' to be a torch.Tensor.")

        wavlm_out = self.wavlm(waveforms)
        emb = wavlm_out.last_hidden_state.mean(dim=1)
        return emb


if __name__ == "__main__":
    # Local imports
    from src.model.block import ResidualStack

    sinc_out_ = 128
    kernel_size_ = 251
    num_blocks_ = 4
    dilation_ = 1
    norm_type_ = "batchnorm"

    sinc_conv_test = nn.Conv1d(
        in_channels=1,
        out_channels=sinc_out_,
        kernel_size=kernel_size_,
        stride=2,
        padding=125
    )

    residual_stack = ResidualStack(
        channels=sinc_out_,
        kernel_size=kernel_size_,
        dilation=dilation_,
        use_se=True,
        num_blocks=num_blocks_,
    )


    class MeanStdPooling(nn.Module):
        """
        Simple pooling layer that concatenates the mean and std
        along the time dimension.
        """

        @staticmethod
        def forward(
                x: Annotated[torch.Tensor, "Input features (B, C, T)"]
        ) -> Annotated[torch.Tensor, "Mean+Std aggregated features (B, C*2, 1)"]:
            mean = x.mean(dim=2)
            std = x.std(dim=2)
            out = torch.cat([mean, std], dim=1)
            return out.unsqueeze(-1)


    pooling_layer_test = MeanStdPooling()

    rawnetx_model_test = RawNetX(
        sinc_conv=sinc_conv_test,
        residual_blocks=residual_stack,
        pooling_layer=pooling_layer_test,
        sinc_out_channels=sinc_out_,
        rawnetx_embedding_dim=128
    )

    x_ = torch.randn(2, 1, 16000)
    rawnetx_output_test = rawnetx_model_test(x_)
    print("RawNetX output shape:", rawnetx_output_test.shape)

    wavlm_model_test = WavLMLarge()
    test_waveforms = torch.randn(2, 16000)
    wavlm_output_test = wavlm_model_test(test_waveforms)
    print("WavLMLarge output shape:", wavlm_output_test.shape)
