# Standard library imports
from typing import Annotated

# Third-party imports
import torch
import torchaudio


class Transform:
    """
    Transform class providing methods for waveform processing.

    This class includes static methods to convert waveforms to mono
    and to resample waveforms from one sample rate to another.

    Methods
    -------
    to_mono(waveform):
        Convert a multi-channel waveform to mono.
    resample(waveform, sr_from, sr_to=16000):
        Resample a waveform from one sample rate to another.

    Examples
    --------
    >>> import torch
    >>> test_waveform = torch.randn(4, 16000)
    >>> mono_waveform = Transform.to_mono(test_waveform)
    >>> mono_waveform.shape
    torch.Size([1, 16000])
    >>> resampled_waveform = Transform.resample(
    ...     mono_waveform, sr_from=16000, sr_to=8000
    ... )
    >>> resampled_waveform.shape  # doctest: +SKIP
    torch.Size([1, 8000])
    """

    @staticmethod
    def to_mono(
            waveform: Annotated[torch.Tensor, "Waveform with shape (channels, samples)"]
    ) -> Annotated[torch.Tensor, "Mono waveform with shape (1, samples)"]:
        """
        Convert a multi-channel waveform to a mono waveform by averaging
        across channels.

        Parameters
        ----------
        waveform : torch.Tensor
            The input waveform. The shape is expected to be
            (channels, samples).

        Returns
        -------
        torch.Tensor
            A mono waveform with shape (1, samples). If the input
            waveform is already mono, it is returned as-is.

        Examples
        --------
        >>> import torch
        >>> waveform_test = torch.randn(2, 16000)
        >>> mono_waveform = Transform.to_mono(waveform_test)
        >>> mono_waveform.shape
        torch.Size([1, 16000])
        """
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Expected 'waveform' to be a torch.Tensor.")

        if waveform.dim() != 2:
            raise ValueError(
                "Expected waveform to have 2 dimensions (channels, samples)."
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    @staticmethod
    def resample(
            waveform: Annotated[torch.Tensor, "Waveform with shape (channels, samples)"],
            sr_from: Annotated[int, "Original sample rate"],
            sr_to: Annotated[int, "Target sample rate"] = 16000
    ) -> Annotated[torch.Tensor, "Resampled waveform with shape (channels, new_samples)"]:
        """
        Resample a waveform from one sample rate to another.

        Parameters
        ----------
        waveform : torch.Tensor
            The input waveform. Shape is expected to be
            (channels, samples).
        sr_from : int
            The current sample rate of the input waveform.
        sr_to : int, optional
            The desired sample rate for output waveform.
            Defaults to 16000.

        Returns
        -------
        torch.Tensor
            Resampled waveform with shape (channels, new_samples).

        Examples
        --------
        >>> import torch
        >>> test_waveform = torch.randn(1, 16000)
        >>> resampled_waveform = Transform.resample(
        ...     test_waveform, sr_from=16000, sr_to=8000
        ... )
        >>> resampled_waveform.shape  # doctest: +SKIP
        torch.Size([1, 8000])
        """
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Expected 'waveform' to be a torch.Tensor.")
        if not isinstance(sr_from, int):
            raise TypeError("Expected 'sr_from' to be an integer.")
        if not isinstance(sr_to, int):
            raise TypeError("Expected 'sr_to' to be an integer.")
        if waveform.dim() != 2:
            raise ValueError(
                "Expected waveform to have 2 dimensions (channels, samples)."
            )

        if sr_from != sr_to:
            resampler = torchaudio.transforms.Resample(sr_from, sr_to)
            waveform = resampler(waveform)
        return waveform


if __name__ == "__main__":
    test_waveform_transform = torch.randn(4, 16000)
    print("Original Waveform Shape:", test_waveform_transform.shape)

    test_waveform_mono_out = Transform.to_mono(test_waveform_transform)
    print("Mono Waveform Shape    :", test_waveform_mono_out.shape)

    sample_rate_from_explicit = 16000
    sample_rate_to_explicit = 8000
    test_waveform_resampled_out = Transform.resample(
        test_waveform_mono_out,
        sr_from=sample_rate_from_explicit,
        sr_to=sample_rate_to_explicit
    )
    print("Resampled Waveform Shape:", test_waveform_resampled_out.shape)
