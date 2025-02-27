# Standard library imports
import os
import glob
from typing import Annotated

# Third-party imports
import torch
import torchaudio
from torch.utils.data import Dataset

# Local imports
from src.preprocess.feature import Segment
from src.preprocess.transformation import Transform


class VoxCeleb1Dataset(Dataset):
    """
    A PyTorch Dataset for loading and processing the VoxCeleb1 audio dataset.

    This class loads .wav files from a given directory structure, filters them
    by speaker ID ranges according to the subset ('train' or 'valid'), applies
    a transformation (e.g., mono conversion, resampling), and slices or pads
    the audio based on a specified segment length.

    Parameters
    ----------
    dataset_root : str
        Path to the dataset root directory.
    transform : Transform
        A transform object that must provide `to_mono` and `resample` methods
        for processing audio waveforms.
    segment_class : type
        A class responsible for slicing or padding the waveforms (e.g., Segment).
    subset : str, optional
        Must be either 'train' or 'valid'. Defaults to 'train'.
    segment_length : float, optional
        Length of segments (in seconds) to be used when slicing or padding
        the waveforms. Defaults to 2.0 seconds.
    sample_rate : int, optional
        Desired sample rate for the processed audio waveforms. Defaults to 16000.

    Attributes
    ----------
    dataset_root : str
        Path to the dataset root directory.
    transform : Transform
        Transform object used to process the waveforms.
    segment_class : type
        Class used for slicing or padding waveforms.
    subset : str
        Either 'train' or 'valid'.
    segment_length : float
        Length (in seconds) for each audio segment.
    sample_rate : int
        Sample rate for the processed audio.
    data_files : list of tuple
        A list of (wav_path, speaker_id) pairs.
    speaker_map : dict
        Maps speaker strings to numeric IDs.
    spk_counter : int
        Counts how many unique speakers have been found.

    Examples
    --------
    >>> dataset = VoxCeleb1Dataset(
    ...     dataset_root="/path/to/vox1",
    ...     transform=Transform(),
    ...     segment_class=Segment,
    ...     segment_length=3.0,
    ...     sample_rate=8000
    ... )
    >>> len(dataset)  # doctest: +SKIP
    100
    """

    def __init__(
            self,
            dataset_root: Annotated[str, "Path to the dataset root directory"],
            transform: Annotated[Transform, "Transform object with audio operations"],
            segment_class: Annotated[type, "Class for slicing/padding waveforms"],
            subset: Annotated[str, "'train' or 'valid' subset selection"] = 'train',
            segment_length: Annotated[float, "Length of segments in seconds"] = 2.0,
            sample_rate: Annotated[int, "Target sample rate for audio"] = 16000
    ) -> None:
        """
        Initialize the VoxCeleb1Dataset with the specified parameters.
        """
        super().__init__()

        # Basic type checks
        if not isinstance(dataset_root, str):
            raise TypeError("Expected 'dataset_root' to be a string.")
        if not isinstance(subset, str):
            raise TypeError("Expected 'subset' to be a string.")
        if not isinstance(segment_length, (float, int)):
            raise TypeError("Expected 'segment_length' to be a float or int.")
        if not isinstance(sample_rate, int):
            raise TypeError("Expected 'sample_rate' to be an int.")

        self.dataset_root = dataset_root
        self.transform = transform
        self.segment_class = segment_class
        self.subset = subset
        self.segment_length = segment_length
        self.sample_rate = sample_rate

        self.data_files = []
        self.speaker_map = {}
        self.spk_counter = 0

        self._collect_wav_files()

        print(
            f"[{self.subset.upper()}] root_dir={dataset_root}, "
            f"#files={len(self.data_files)}, "
            f"#speakers={len(self.speaker_map)}"
        )

    def _collect_wav_files(self) -> None:
        """
        Collect all .wav file paths for the given subset ('train' or 'valid')
        and store them along with speaker IDs.

        Raises
        ------
        ValueError
            If the 'subset' is not 'train' or 'valid'.
        """
        id_folders = sorted(glob.glob(os.path.join(self.dataset_root, "id*")))

        if self.subset == 'train':
            min_id, max_id = 10001, 11241
        elif self.subset == 'valid':
            min_id, max_id = 11242, 11251
        else:
            raise ValueError("The 'subset' parameter must be either 'train' or 'valid'.")

        for spk_path in id_folders:
            spk_name = os.path.basename(spk_path)
            if not spk_name.startswith("id"):
                continue

            try:
                spk_num = int(spk_name[2:])
            except ValueError:
                continue

            if not (min_id <= spk_num <= max_id):
                continue

            spk_str = str(spk_num)
            if spk_str not in self.speaker_map:
                self.speaker_map[spk_str] = self.spk_counter
                self.spk_counter += 1
            spk_id = self.speaker_map[spk_str]

            wav_paths = glob.glob(
                os.path.join(spk_path, "**", "*.wav"),
                recursive=True
            )
            for wav_file in wav_paths:
                self.data_files.append((wav_file, spk_id))

    def __len__(self) -> Annotated[int, "Number of audio files in the dataset"]:
        """
        Return the number of audio files in the dataset.

        Returns
        -------
        int
            The total number of audio files in this dataset.

        Examples
        --------
        >>> dataset = VoxCeleb1Dataset(
        ...     dataset_root="/path/to/vox1",
        ...     transform=Transform(),
        ...     segment_class=Segment
        ... )
        >>> len(dataset)  # doctest: +SKIP
        100
        """
        return len(self.data_files)

    def __getitem__(
            self,
            idx: Annotated[int, "Index of the dataset item"]
    ) -> Annotated[tuple, "Tuple of (processed waveform, speaker ID)"]:
        """
        Retrieve and process a single waveform and its associated speaker ID.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            (waveform_processed, speaker_id):

            - waveform_processed is a 1D tensor if transformed to mono.
            - speaker_id is the integer speaker ID associated with this waveform.

        Examples
        --------
        >>> dataset = VoxCeleb1Dataset(
        ...     dataset_root="/path/to/vox1",
        ...     transform=Transform(),
        ...     segment_class=Segment
        ... )
        >>> waveform_test, speaker_id_test = dataset[0]  # doctest: +SKIP
        >>> waveform_test.shape  # doctest: +SKIP
        torch.Size([32000])
        """
        if not isinstance(idx, int):
            raise TypeError("Index 'idx' must be an integer.")

        wav_path, speaker_id = self.data_files[idx]
        waveform, sr = torchaudio.load(wav_path)

        # Convert to mono and resample
        waveform = self.transform.to_mono(waveform)
        waveform = self.transform.resample(
            waveform,
            sr_from=sr,
            sr_to=self.sample_rate
        )

        seg = self.segment_class(waveform)
        seg_samples = int(self.segment_length * self.sample_rate)

        total_samples = waveform.shape[1]
        if total_samples > seg_samples:
            seg.slice(seg_samples)
        else:
            seg.pad(seg_samples)

        waveform_processed = seg.waveform.squeeze(0)
        return waveform_processed, speaker_id


class Collate:
    """
    Collation function for batching samples of (waveform, label).

    This class provides a callable that can be used as the `collate_fn`
    in a PyTorch DataLoader. It stacks multiple waveforms into a single tensor
    and collects their labels into another tensor.

    Methods
    -------
    __call__(batch)
        Collates a list of (waveform, label) pairs into batch tensors.

    Examples
    --------
    >>> collate_fn = Collate()
    >>> waveforms, labels = collate_fn([
    ...     (torch.randn(16000), 0),
    ...     (torch.randn(16000), 1)
    ... ])
    >>> waveforms.shape
    torch.Size([2, 16000])
    >>> labels
    tensor([0, 1])
    """

    def __call__(
            self,
            batch: Annotated[list, "List of (waveform, label) pairs"]
    ) -> Annotated[tuple, "Tuple of (batched waveforms, batched labels)"]:
        """
        Collate a list of (waveform, label) pairs into batch tensors.

        Parameters
        ----------
        batch : list
            A list of tuples (waveform, label).

        Returns
        -------
        tuple
            (waveforms, labels):

            - waveforms is a 2D or 3D stacked tensor of shape (batch_size, ...).
            - labels is a 1D tensor of speaker IDs.

        Examples
        --------
        >>> collate_fn = Collate()
        >>> waveforms_test, labels_test = collate_fn([
        ...     (torch.randn(16000), 0),
        ...     (torch.randn(16000), 1)
        ... ])
        >>> waveforms_test.shape
        torch.Size([2, 16000])
        >>> labels_test
        tensor([0, 1])
        """
        waveforms, labels = [], []
        for item in batch:
            if (
                    not isinstance(item, tuple) or
                    len(item) != 2
            ):
                raise ValueError(
                    "Each batch item must be a tuple (waveform, label)."
                )
            wf, spk = item
            waveforms.append(wf)
            labels.append(spk)

        waveforms = torch.stack(waveforms)
        labels = torch.tensor(labels, dtype=torch.long)
        return waveforms, labels


if __name__ == "__main__":
    # Third-party imports
    from torch.utils.data import DataLoader

    transform_for_test = Transform()
    segment_class_for_test = Segment

    dataset_manager_test = VoxCeleb1Dataset(
        dataset_root=".data/dataset/train/VoxCeleb1/dev/vox1_dev_wav/wav",
        transform=transform_for_test,
        segment_class=segment_class_for_test,
        segment_length=3.0,
        sample_rate=8000
    )

    print("Dataset length:", len(dataset_manager_test))

    collate_fn_test = Collate()
    data_loader_test = DataLoader(
        dataset_manager_test,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn_test
    )

    for test_idx, (waveforms_batch, labels_batch) in enumerate(data_loader_test):
        print(f"Batch #{test_idx}:")
        print("  Waveforms:", waveforms_batch)
        print("  Labels:", labels_batch)
        break
