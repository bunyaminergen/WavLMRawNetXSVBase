# Standard library imports
from typing import Annotated

# Third-party imports
import torch
import torch.nn as nn


class Fusion(nn.Module):
    """
    Fusion module that combines embeddings from two different models
    (e.g., WavLM and RawNetX) into a single fused embedding.

    This class expects two models: a WavLM-based model that takes
    a waveform of shape (B, T) and returns an embedding of size
    `wavlm_embedding_dim`, and a RawNetX-based model that takes a
    waveform of shape (B, 1, T) and returns an embedding of size
    `rawnetx_embedding_dim`. The embeddings are concatenated and
    passed through a fully connected layer, producing an output
    embedding of dimension `fusion_embedding_dim`.

    Parameters
    ----------
    wavlm_model : nn.Module
        A model that accepts waveforms of shape (batch_size, samples)
        and produces embeddings of dimension `wavlm_embedding_dim`.
    rawnetx_model : nn.Module
        A model that accepts waveforms of shape (batch_size, 1, samples)
        and produces embeddings of dimension `rawnetx_embedding_dim`.
    wavlm_embedding_dim : int, optional
        Dimensionality of the embeddings produced by the `wavlm_model`.
        Defaults to 1024.
    rawnetx_embedding_dim : int, optional
        Dimensionality of the embeddings produced by the `rawnetx_model`.
        Defaults to 256.
    fusion_embedding_dim : int, optional
        Output dimensionality of the fused embeddings. Defaults to 256.

    Attributes
    ----------
    wavlm_model : nn.Module
        The model for WavLM-based embeddings.
    rawnetx_model : nn.Module
        The model for RawNetX-based embeddings.
    fusion_fc : nn.Linear
        Fully connected layer for merging and projecting embeddings.
    act : nn.ReLU
        Activation function applied to the fused embeddings.

    Examples
    --------
    >>> import torch
    >>> class MockWavLM(nn.Module):
    ...     @staticmethod
    ...     def forward(waveforms_test):
    ...         return torch.randn(waveforms_test.size(0), 512)
    ...
    >>> class MockRawNetX(nn.Module):
    ...     @staticmethod
    ...     def forward(x):
    ...         return torch.randn(x.size(0), 128)
    ...
    >>> wavlm_mock = MockWavLM()
    >>> rawnetx_mock = MockRawNetX()
    >>> fusion_model = Fusion(
    ...     wavlm_model=wavlm_mock,
    ...     rawnetx_model=rawnetx_mock,
    ...     wavlm_embedding_dim=512,
    ...     rawnetx_embedding_dim=128,
    ... )
    >>> waveforms = torch.randn(2, 16000)
    >>> fused_output = fusion_model(waveforms)
    >>> fused_output.shape
    torch.Size([2, 256])
    """

    def __init__(
            self,
            wavlm_model: Annotated[nn.Module, "Model returning WavLM embeddings"],
            rawnetx_model: Annotated[nn.Module, "Model returning RawNetX embeddings"],
            wavlm_embedding_dim: Annotated[int, "Dimension of WavLM embeddings"] = 1024,
            rawnetx_embedding_dim: Annotated[int, "Dimension of RawNetX embeddings"] = 256,
            fusion_embedding_dim: Annotated[int, "Dimension of fused embeddings"] = 256
    ) -> None:
        """
        Initialize the Fusion module by creating a linear layer that
        merges the two embeddings into a unified embedding of size
        `fusion_embedding_dim`.
        """
        super().__init__()

        if not isinstance(wavlm_model, nn.Module):
            raise TypeError("Expected 'wavlm_model' to be an nn.Module.")
        if not isinstance(rawnetx_model, nn.Module):
            raise TypeError("Expected 'rawnetx_model' to be an nn.Module.")
        if not all(isinstance(dim, int) for dim in [
            wavlm_embedding_dim, rawnetx_embedding_dim, fusion_embedding_dim
        ]):
            raise TypeError("Embedding dimensions must be integers.")

        self.wavlm_model = wavlm_model
        self.rawnetx_model = rawnetx_model

        self.fusion_fc = nn.Linear(
            wavlm_embedding_dim + rawnetx_embedding_dim,
            fusion_embedding_dim
        )
        self.act = nn.ReLU()

    def forward(
            self,
            waveforms: Annotated[torch.Tensor, "(B, T) waveform inputs"]
    ) -> Annotated[torch.Tensor, "(B, fusion_embedding_dim) fused embeddings"]:
        """
        Forward pass for the Fusion model. Feeds the same waveform
        data to both the WavLM and RawNetX models, concatenates
        their embeddings, and applies a fully connected layer and
        activation to produce the final fused embedding.

        Parameters
        ----------
        waveforms : torch.Tensor
            A batch of waveforms of shape (batch_size, samples).

        Returns
        -------
        torch.Tensor
            A tensor of fused embeddings of shape
            (batch_size, fusion_embedding_dim).

        Examples
        --------
        >>> import torch
        >>> class MockWavLM(nn.Module):
        ...     @staticmethod
        ...     def forward(w):
        ...         return torch.randn(w.size(0), 512)
        ...
        >>> class MockRawNetX(nn.Module):
        ...     @staticmethod
        ...     def forward(w):
        ...         return torch.randn(w.size(0), 128)
        ...
        >>> wavlm_mock = MockWavLM()
        >>> rawnetx_mock = MockRawNetX()
        >>> fusion = Fusion(
        ...     wavlm_model=wavlm_mock,
        ...     rawnetx_model=rawnetx_mock,
        ...     wavlm_embedding_dim=512,
        ...     rawnetx_embedding_dim=128,
        ... )
        >>> test_waveforms = torch.randn(2, 16000)
        >>> fused_output = fusion(test_waveforms)
        >>> fused_output.shape
        torch.Size([2, 256])
        """
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("Expected 'waveforms' to be a torch.Tensor.")

        # Get WavLM embeddings (expect shape (B, wavlm_embedding_dim))
        wavlm_emb = self.wavlm_model(waveforms)

        # For RawNetX, unsqueeze channel dimension -> shape (B, 1, T)
        rawnetx_emb = self.rawnetx_model(waveforms.unsqueeze(1))

        # Concatenate along the embedding dimension
        fused = torch.cat([wavlm_emb, rawnetx_emb], dim=1)

        # Fully connected layer and activation
        fused = self.fusion_fc(fused)
        fused = self.act(fused)
        return fused


if __name__ == "__main__":
    class MockWavLMTest(nn.Module):
        """
        Mock WavLM model for testing Fusion.
        Generates random embeddings of shape (B, output_dim).
        """

        def __init__(self, output_dim: int):
            super().__init__()
            self.output_dim = output_dim

        @staticmethod
        def forward(
                self,
                waveforms: Annotated[torch.Tensor, "(B, T) waveform inputs"]
        ) -> Annotated[torch.Tensor, "(B, output_dim) embeddings"]:
            batch_size = waveforms.size(0)
            return torch.randn(batch_size, self.output_dim)


    class MockRawNetXTest(nn.Module):
        """
        Mock RawNetX model for testing Fusion.
        Generates random embeddings of shape (B, output_dim).
        """

        def __init__(self, output_dim: int):
            super().__init__()
            self.output_dim = output_dim

        def forward(
                self,
                x: Annotated[torch.Tensor, "(B, 1, T) waveform inputs"]
        ) -> Annotated[torch.Tensor, "(B, output_dim) embeddings"]:
            batch_size = x.size(0)
            return torch.randn(batch_size, self.output_dim)


    # Instantiate mock models
    mock_wavlm_test_model = MockWavLMTest(output_dim=512)
    mock_rawnetx_test_model = MockRawNetXTest(output_dim=128)

    # Instantiate Fusion
    fusion_model_test = Fusion(
        wavlm_model=mock_wavlm_test_model,
        rawnetx_model=mock_rawnetx_test_model,
        wavlm_embedding_dim=512,
        rawnetx_embedding_dim=128,
    )

    waveforms_test_fusion = torch.randn(2, 16000)
    fusion_output_test = fusion_model_test(waveforms_test_fusion)

    print("Fusion Input shape:", waveforms_test_fusion.shape)
    print("Fusion Output shape:", fusion_output_test.shape)
