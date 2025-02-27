# Standard library imports
import os
from typing import Annotated

# Third-party imports
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F

# Local imports
from src.evaluate.metric import Metrics
from src.preprocess.transformation import Transform


class Tester:
    """
    Tester class for evaluating a speaker verification model.

    This class loads pairs of audio files specified in a text file and
    computes cosine similarity scores between their embeddings. It then
    calculates metrics such as EER and minDCF.

    Parameters
    ----------
    model_sv : torch.nn.Module
        A speaker verification model that takes waveforms as input and
        returns embeddings.
    transform : Transform
        A transform object that should provide `to_mono` and `resample`
        methods for audio processing.
    metrics : Metrics
        An object offering `eer` and `mindcf` metrics.
    test_wav_root : str
        Root directory containing test audio files.
    test_pairs_txt : str
        Path to the text file containing test pairs (and labels).
    device : str, optional
        Device to use for evaluation (e.g., "cuda" or "cpu").
        Defaults to "cuda".
    sample_rate : int, optional
        Sampling rate to use when resampling audio. Defaults to 16000.

    Attributes
    ----------
    model_sv : torch.nn.Module
        The speaker verification model used for generating embeddings.
    transform : Transform
        Transformation instance for audio preprocessing.
    metrics : Metrics
        Provides EER and minDCF metrics.
    test_wav_root : str
        Directory containing test .wav files.
    test_pairs_txt : str
        File containing lines with format "<label> <path1> <path2>".
    device : str
        The computation device (e.g., "cuda").
    sample_rate : int
        Sampling rate for audio.

    Examples
    --------
    >>> # my_model = MyModel().eval()   # doctest: +SKIP
    >>> # transform = MyTransform()     # doctest: +SKIP
    >>> # metrics = MyMetrics()         # doctest: +SKIP
    >>> # tester = Tester(
    ... #     model_sv=my_model,
    ... #     transform=transform,
    ... #     metrics=metrics,
    ... #     test_wav_root="path/to/test/wavs",
    ... #     test_pairs_txt="path/to/test_pairs.txt",
    ... #     device="cpu",
    ... #     sample_rate=16000
    ... # )
    >>> # tester.run()  # doctest: +SKIP
    """

    def __init__(
            self,
            model_sv: Annotated[torch.nn.Module, "Speaker verification model"],
            transform: Annotated[Transform, "Transform for audio (to_mono, resample)"],
            metrics: Annotated[Metrics, "Metrics for EER and minDCF"],
            test_wav_root: Annotated[str, "Root directory for test WAV files"],
            test_pairs_txt: Annotated[str, "Path to test pairs file"],
            device: Annotated[str, "Computation device"] = "cuda",
            sample_rate: Annotated[int, "Sampling rate"] = 16000
    ) -> None:
        """
        Initialize the Tester with a model, transform, metrics, paths,
        device, and sample rate.

        Raises
        ------
        TypeError
            If any of the provided parameters have incorrect types.
        """
        if not isinstance(model_sv, torch.nn.Module):
            raise TypeError("Expected 'model_sv' to be a 'torch.nn.Module'.")
        if not isinstance(transform, Transform):
            raise TypeError("Expected 'transform' to be an instance of 'Transform'.")
        if not isinstance(metrics, Metrics):
            raise TypeError("Expected 'metrics' to be an instance of 'Metrics'.")
        if not isinstance(test_wav_root, str):
            raise TypeError("Expected 'test_wav_root' to be a string.")
        if not isinstance(test_pairs_txt, str):
            raise TypeError("Expected 'test_pairs_txt' to be a string.")
        if not isinstance(device, str):
            raise TypeError("Expected 'device' to be a string.")
        if not isinstance(sample_rate, int):
            raise TypeError("Expected 'sample_rate' to be an integer.")

        self.model_sv = model_sv
        self.transform = transform
        self.metrics = metrics
        self.test_wav_root = test_wav_root
        self.test_pairs_txt = test_pairs_txt
        self.device = device
        self.sample_rate = sample_rate

        self.model_sv.eval()

    def run(self) -> None:
        """
        Run the evaluation of the speaker verification model on test pairs.

        Reads the test pairs file, processes each audio pair, computes
        embeddings, calculates cosine similarity scores, and then derives
        EER and minDCF metrics.

        Returns
        -------
        None

        Examples
        --------
        >>> # tester = Tester(...)  # doctest: +SKIP
        >>> # tester.run()          # doctest: +SKIP
        """
        test_labels = []
        test_scores = []

        with open(self.test_pairs_txt) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            label_str, path1, path2 = parts[0], parts[1], parts[2]

            try:
                label = int(label_str)
            except ValueError:
                continue

            full_path1 = os.path.join(self.test_wav_root, path1)
            if not os.path.isfile(full_path1):
                continue

            full_path2 = os.path.join(self.test_wav_root, path2)
            if not os.path.isfile(full_path2):
                continue

            waveform1, sr1 = torchaudio.load(full_path1)
            waveform1 = self.transform.to_mono(waveform1)
            waveform1 = self.transform.resample(waveform1, sr_from=sr1, sr_to=self.sample_rate)
            waveform1 = waveform1.squeeze(0).to(self.device)

            waveform2, sr2 = torchaudio.load(full_path2)
            waveform2 = self.transform.to_mono(waveform2)
            waveform2 = self.transform.resample(waveform2, sr_from=sr2, sr_to=self.sample_rate)
            waveform2 = waveform2.squeeze(0).to(self.device)

            with torch.no_grad():
                emb1 = self.model_sv(waveform1.unsqueeze(0))
                emb2 = self.model_sv(waveform2.unsqueeze(0))

            emb1_n = F.normalize(emb1)
            emb2_n = F.normalize(emb2)
            score = torch.sum(emb1_n * emb2_n, dim=1).item()

            test_labels.append(label)
            test_scores.append(score)

        labels_array = np.array(test_labels)
        scores_array = np.array(test_scores)

        eer_test = self.metrics.eer(scores_array, labels_array)
        min_dcf_test = self.metrics.mindcf(scores_array, labels_array)

        print(f"Test EER  : {eer_test * 100:.2f}%")
        print(f"Test minDCF: {min_dcf_test:.4f}")


if __name__ == "__main__":
    # Local imports
    from src.model.backbone import RawNetX, WavLMLarge
    from src.model.block import ResidualStack
    from src.model.convolution import SincConv
    from src.model.fusion import Fusion
    from src.model.pooling import AttentiveStatsPooling

    model_checkpoint_path = ".model/WavLMRawNetXSVBase.pt"
    test_wav_root_ = ".data/dataset/train/VoxCeleb1/test/vox1_test_wav/wav"
    test_pairs_txt_ = ".data/dataset/train/VoxCeleb1/test/veri_test2.txt"

    sample_rate_ = 16000
    sinc_out_ = 128
    kernel_size_ = 251
    num_blocks_ = 4
    dilation_ = 1
    norm_type_ = "batchnorm"
    rawnetx_embedding_dim_ = 256
    wavlm_embedding_dim_ = 1024
    fusion_embedding_dim_ = 256

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sinc_conv_ = SincConv(
        out_channels=sinc_out_,
        kernel_size=kernel_size_,
        sample_rate=sample_rate_,
        padding=kernel_size_ // 2
    )
    if norm_type_ == "instancenorm":
        first_norm_ = torch.nn.InstanceNorm1d(sinc_out_, affine=True)
    else:
        first_norm_ = torch.nn.BatchNorm1d(sinc_out_)
    first_act_ = torch.nn.Mish()

    pooling_layer_ = AttentiveStatsPooling(in_dim=sinc_out_)
    residual_stack = ResidualStack(
        channels=sinc_out_,
        kernel_size=kernel_size_,
        dilation=dilation_,
        use_se=True,
        num_blocks=num_blocks_,
    )
    rawnetx_model_ = RawNetX(
        sinc_conv=sinc_conv_,
        residual_blocks=residual_stack,
        pooling_layer=pooling_layer_,
        rawnetx_embedding_dim=rawnetx_embedding_dim_,
        sinc_out_channels=sinc_out_,
    )

    wavlm_model_ = WavLMLarge()

    wavlm_rawnetx_sv_base = Fusion(
        wavlm_model=wavlm_model_,
        rawnetx_model=rawnetx_model_,
        wavlm_embedding_dim=wavlm_embedding_dim_,
        rawnetx_embedding_dim=rawnetx_embedding_dim_,
        fusion_embedding_dim=fusion_embedding_dim_
    ).to(device_)

    checkpoint_ = torch.load(model_checkpoint_path, map_location=device_)
    wavlm_rawnetx_sv_base.load_state_dict(checkpoint_["model_state"], strict=False)
    wavlm_rawnetx_sv_base.eval()

    transform_ = Transform()
    metrics_ = Metrics()

    tester_ = Tester(
        model_sv=wavlm_rawnetx_sv_base,
        transform=transform_,
        metrics=metrics_,
        test_wav_root=test_wav_root_,
        test_pairs_txt=test_pairs_txt_,
        device="cpu",
        sample_rate=sample_rate_
    )
    tester_.run()
