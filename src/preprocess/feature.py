# Standard library imports
import random
from typing import Annotated

# Third-party imports
import torch


class Segment:
    """
    Segment class for slicing and padding waveforms.

    This class provides methods to slice a waveform to a specified number
    of samples, or to pad a waveform up to a specified number of samples.
    It ensures consistent segment length across audio samples by either
    trimming or zero-padding.

    Parameters
    ----------
    waveform : torch.Tensor
        A waveform tensor of shape (channels, samples). The first dimension
        corresponds to the number of channels, and the second dimension
        corresponds to the number of samples.

    Examples
    --------
    Create an instance of Segment and slice/pad the waveform:

    >>> waveform = torch.randn(2, 2000)
    >>> segment = Segment(waveform=waveform)
    >>> sliced_waveform = segment.slice(segment_samples=600)
    >>> padded_waveform = segment.pad(segment_samples=2500)
    """

    def __init__(
            self,
            waveform: Annotated[torch.Tensor, "Waveform (channels, samples)"]
    ) -> None:
        """
        Initialize the Segment class with a waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            A waveform tensor of shape (channels, samples). The first
            dimension corresponds to the number of channels, and the
            second dimension corresponds to the number of samples.

        Raises
        ------
        TypeError
            If 'waveform' is not a torch.Tensor.
        """
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Expected 'waveform' to be a torch.Tensor.")
        self.waveform = waveform

    def slice(
            self,
            segment_samples: Annotated[int, "Number of samples to slice to"]
    ) -> Annotated[torch.Tensor, "Sliced waveform of shape (channels, segment_samples)"]:
        """
        Slice the waveform to the specified number of samples.

        If the current number of samples exceeds `segment_samples`, a random
        start point is selected to obtain a slice of length `segment_samples`.
        Otherwise, the waveform remains unchanged.

        Parameters
        ----------
        segment_samples : int
            Number of samples to keep in the slice.

        Returns
        -------
        torch.Tensor
            The sliced or unchanged waveform.

        Examples
        --------
        >>> import torch
        >>> waveform = torch.randn(1, 2000)
        >>> segment = Segment(waveform=waveform)
        >>> sliced = segment.slice(segment_samples=600)
        >>> sliced.shape
        torch.Size([1, 600])
        """
        if not isinstance(segment_samples, int):
            raise TypeError("Expected 'segment_samples' to be an integer.")

        total_samples = self.waveform.shape[1]
        if total_samples > segment_samples:
            max_start = total_samples - segment_samples
            start = random.randint(0, max_start)
            end = start + segment_samples
            self.waveform = self.waveform[:, start:end]
        return self.waveform

    def pad(
            self,
            segment_samples: Annotated[int, "Number of samples to pad to"]
    ) -> Annotated[torch.Tensor, "Padded waveform of shape (channels, segment_samples)"]:
        """
        Pad the waveform up to the specified number of samples.

        If the current number of samples is less than `segment_samples`,
        zero-padding is added to the end of the waveform to reach the
        specified number of samples. Otherwise, the waveform remains
        unchanged.

        Parameters
        ----------
        segment_samples : int
            Number of samples to pad up to.

        Returns
        -------
        torch.Tensor
            The padded or unchanged waveform.

        Examples
        --------
        >>> waveform = torch.randn(1, 800)
        >>> segment = Segment(waveform=waveform)
        >>> padded = segment.pad(segment_samples=1000)
        >>> padded.shape
        torch.Size([1, 1000])
        """
        if not isinstance(segment_samples, int):
            raise TypeError("Expected 'segment_samples' to be an integer.")

        total_samples = self.waveform.shape[1]
        if total_samples < segment_samples:
            pad_size = segment_samples - total_samples
            pad = torch.zeros(
                (self.waveform.shape[0], pad_size),
                dtype=self.waveform.dtype,
                device=self.waveform.device
            )
            self.waveform = torch.cat([self.waveform, pad], dim=1)
        return self.waveform


if __name__ == "__main__":
    segment_test_waveform = torch.randn(2, 2000)

    segment_instance_test = Segment(waveform=segment_test_waveform)

    slice_output_test = segment_instance_test.slice(segment_samples=600)
    print("Slice Shape:", slice_output_test.shape)

    pad_output_test = segment_instance_test.pad(segment_samples=2500)
    print("Pad Shape:", pad_output_test.shape)
