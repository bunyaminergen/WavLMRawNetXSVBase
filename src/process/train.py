# Standard library imports
import os
from typing import Annotated

# Third-party imports
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports
from src.evaluate.metric import Metrics


class Trainer:
    """
    Trainer class for speech verification tasks.

    This class handles the training, validation, and model saving workflow.
    It utilizes a speaker verification model, a loss function (AMSoftmax),
    and an optimizer to perform the training routine. Validation is done by
    computing embeddings on a validation set and deriving metrics such as EER
    and minDCF.

    Parameters
    ----------
    train_loader : DataLoader
        PyTorch DataLoader for the training set.
    val_loader : DataLoader
        PyTorch DataLoader for the validation set.
    model_sv : torch.nn.Module
        The speaker verification model to be trained.
    amsoftmax : torch.nn.Module
        The AMSoftmax loss module used for classification loss.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    metrics : Metrics
        Metrics class instance providing EER and minDCF calculation.
    device : str, optional
        Device to use for computation. Defaults to "cuda".
    epochs : int, optional
        Number of training epochs. Defaults to 3.
    save_model_path : str, optional
        File path to save the trained model. Defaults to
        ".model/WavLMRawNetXSVBase.pt".

    Attributes
    ----------
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    model_sv : torch.nn.Module
        Speaker verification model.
    amsoftmax : torch.nn.Module
        AMSoftmax loss module.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    metrics : Metrics
        Provides methods for EER and minDCF calculation.
    device : str
        Device used for computation (e.g., "cuda" or "cpu").
    epochs : int
        Number of training epochs.
    save_model_path : str
        Path where the model will be saved after training.

    Examples
    --------
    Suppose you have already defined DataLoader objects for training and
    validation, a speaker verification model `my_model`, an `AMSoftmaxLoss`
    instance, an optimizer, and a metrics object:

    >>> from torch.utils.data import DataLoader
    >>> train_loader = DataLoader(...)  # doctest: +SKIP
    >>> val_loader = DataLoader(...)    # doctest: +SKIP
    >>> model_sv = ...                  # doctest: +SKIP
    >>> amsoftmax = ...                 # doctest: +SKIP
    >>> optimizer = ...                 # doctest: +SKIP
    >>> metrics = ...                   # doctest: +SKIP
    >>> trainer = Trainer(
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     model_sv=model_sv,
    ...     amsoftmax=amsoftmax,
    ...     optimizer=optimizer,
    ...     metrics=metrics,
    ...     device="cpu",
    ...     epochs=1,
    ...     save_model_path="model.pt"
    ... )
    >>> trainer.run()  # doctest: +SKIP
    """

    def __init__(
            self,
            train_loader: Annotated[DataLoader, "PyTorch DataLoader for training"],
            val_loader: Annotated[DataLoader, "PyTorch DataLoader for validation"],
            model_sv: Annotated[torch.nn.Module, "Speaker verification model"],
            amsoftmax: Annotated[torch.nn.Module, "AMSoftmax loss module"],
            optimizer: Annotated[torch.optim.Optimizer, "Optimizer for training"],
            metrics: Annotated[Metrics, "Metrics object for EER & minDCF"],
            device: Annotated[str, "Computation device"] = "cuda",
            epochs: Annotated[int, "Number of training epochs"] = 3,
            save_model_path: Annotated[
                str,
                "Path to save the trained model"
            ] = ".model/WavLMRawNetXSVBase.pt"
    ) -> None:
        """
        Initialize the Trainer with provided DataLoaders, model,
        loss function, optimizer, metrics, device, epochs, and
        model save path.

        Raises
        ------
        TypeError
            If an input parameter is not of the expected type.
        """
        if not isinstance(train_loader, DataLoader):
            raise TypeError("Expected 'train_loader' to be a 'DataLoader' instance.")
        if not isinstance(val_loader, DataLoader):
            raise TypeError("Expected 'val_loader' to be a 'DataLoader' instance.")
        if not isinstance(metrics, Metrics):
            raise TypeError("Expected 'metrics' to be an instance of 'Metrics'.")
        if not isinstance(device, str):
            raise TypeError("Expected 'device' to be a string.")
        if not isinstance(epochs, int):
            raise TypeError("Expected 'epochs' to be an integer.")
        if not isinstance(save_model_path, str):
            raise TypeError("Expected 'save_model_path' to be a string.")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_sv = model_sv
        self.amsoftmax = amsoftmax
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device
        self.epochs = epochs
        self.save_model_path = save_model_path

    def epoch(
            self,
            epoch: Annotated[int, "Current epoch index (0-based)"]
    ) -> Annotated[float, "Average training loss for this epoch"]:
        """
        Conduct one full pass (epoch) over the training set.

        Parameters
        ----------
        epoch : int
            The current epoch index (0-based).

        Returns
        -------
        float
            The average training loss for this epoch.

        Examples
        --------
        >>> trainer = Trainer(...)  # doctest: +SKIP
        >>> avg_loss_test = trainer.epoch(0)  # doctest: +SKIP
        >>> print(avg_loss_test)  # doctest: +SKIP
        """
        if not isinstance(epoch, int):
            raise TypeError("Expected 'epoch' to be an integer.")

        self.model_sv.train()
        self.amsoftmax.train()

        total_loss = 0.0
        for waveforms, labels in self.train_loader:
            if not isinstance(waveforms, torch.Tensor):
                raise TypeError("Expected waveforms to be a torch.Tensor.")
            if not isinstance(labels, torch.Tensor):
                raise TypeError("Expected labels to be a torch.Tensor.")

            waveforms = waveforms.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            emb = self.model_sv(waveforms)
            loss, _ = self.amsoftmax(emb, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        print(f"[TRAIN] Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(
            self,
            epoch: Annotated[int, "Current epoch index (0-based)"]
    ) -> None:
        """
        Evaluate the model on the validation set.

        Embeddings are computed for each batch in the validation set,
        normalized, and pairwise cosine similarities are calculated.
        The EER and minDCF metrics are then computed and displayed.

        Parameters
        ----------
        epoch : int
            The current epoch index (0-based).

        Returns
        -------
        None

        Examples
        --------
        >>> trainer = Trainer(...)  # doctest: +SKIP
        >>> trainer.validate(0)  # doctest: +SKIP
        """
        if not isinstance(epoch, int):
            raise TypeError("Expected 'epoch' to be an integer.")

        self.model_sv.eval()
        self.amsoftmax.eval()

        val_embeddings = []
        val_speaker_ids = []

        with torch.no_grad():
            for waveforms_val, labels_val in self.val_loader:
                if not isinstance(waveforms_val, torch.Tensor):
                    raise TypeError("Expected waveforms_val to be a torch.Tensor.")
                if not isinstance(labels_val, torch.Tensor):
                    raise TypeError("Expected labels_val to be a torch.Tensor.")

                waveforms_val = waveforms_val.to(self.device, dtype=torch.float32)
                emb_val = self.model_sv(waveforms_val)
                val_embeddings.append(emb_val.cpu())
                val_speaker_ids.extend(labels_val.tolist())

        val_embeddings = torch.cat(val_embeddings)
        val_speaker_ids = np.array(val_speaker_ids)

        scores_val = []
        labels_bin_val = []
        val_embeddings_norm = F.normalize(val_embeddings)
        d_val = val_embeddings_norm.size(0)

        for i in range(d_val):
            for j in range(i + 1, d_val):
                same_spk = 1 if val_speaker_ids[i] == val_speaker_ids[j] else 0
                labels_bin_val.append(same_spk)
                score = torch.sum(
                    val_embeddings_norm[i] * val_embeddings_norm[j]
                ).item()
                scores_val.append(score)

        scores_val = np.array(scores_val)
        labels_bin_val = np.array(labels_bin_val)

        eer_val = self.metrics.eer(scores_val, labels_bin_val)
        min_dcf_val = self.metrics.mindcf(scores_val, labels_bin_val)
        print(
            f"[VALID] => Epoch {epoch + 1}/{self.epochs}, "
            f"EER: {eer_val * 100:.2f}%, minDCF: {min_dcf_val:.3f}\n"
        )

    def run(self) -> None:
        """
        Run the full training and validation process for the configured
        number of epochs. After training, saves the model and AMSoftmax
        state to the specified file path.

        Returns
        -------
        None

        Examples
        --------
        >>> trainer = Trainer(...)  # doctest: +SKIP
        >>> trainer.run()  # doctest: +SKIP
        """
        for epoch in range(self.epochs):
            self.epoch(epoch)
            self.validate(epoch)

        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)
        checkpoint = {
            "model_state": self.model_sv.state_dict(),
            "amsoftmax_state": self.amsoftmax.state_dict(),
            "model_name": "WavLMRawNetXSVBase",
        }
        torch.save(checkpoint, self.save_model_path)
        print(f"Model saved at: {self.save_model_path}")


if __name__ == "__main__":
    # Third-party imports
    from torch.utils.data import Subset

    # Local imports
    from src.model.fusion import Fusion
    from src.model.loss import AMSoftmaxLoss
    from src.model.block import ResidualStack
    from src.model.convolution import SincConv
    from src.preprocess.feature import Segment
    from src.model.backbone import RawNetX, WavLMLarge
    from src.model.pooling import AttentiveStatsPooling
    from src.preprocess.transformation import Transform
    from src.utils.data.manager import VoxCeleb1Dataset, Collate

    dataset_root_path_ = ".data/dataset/train/VoxCeleb1/dev/vox1_dev_wav/wav"
    test_wav_root_ = ".data/dataset/train/VoxCeleb1/test/vox1_test_wav/wav"
    test_pairs_txt_ = ".data/dataset/train/VoxCeleb1/test/veri_test2.txt"
    model_save_path_ = ".model/WavLMRawNetXSVBaseTest.pt"
    wavlm_dir_ = "microsoft/wavlm-large"

    sample_rate_ = 16000
    segment_length_ = 2.0
    rawnetx_embedding_dim_ = 256
    wavlm_embedding_dim_ = 1024
    fusion_embedding_dim_ = 256
    batch_size_ = 4
    epochs_ = 2
    learning_rate_ = 1e-5
    amsoftmax_margin_ = 0.3
    amsoftmax_scale_ = 30.0
    train_shuffle_ = True
    val_shuffle_ = False

    sinc_out_ = 128
    kernel_size_ = 251
    num_blocks_ = 4
    dilation_ = 1
    norm_type_ = "batchnorm"

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device (main):", device_)

    transform_ = Transform()
    collate_ = Collate()

    train_dataset_ = VoxCeleb1Dataset(
        dataset_root=dataset_root_path_,
        segment_length=segment_length_,
        sample_rate=sample_rate_,
        transform=transform_,
        segment_class=Segment
    )
    val_dataset_ = VoxCeleb1Dataset(
        dataset_root=dataset_root_path_,
        subset='valid',
        segment_length=segment_length_,
        sample_rate=sample_rate_,
        transform=transform_,
        segment_class=Segment
    )

    train_subset_indices = range(4)
    val_subset_indices = range(2)

    train_dataset_ = Subset(train_dataset_, train_subset_indices)
    val_dataset_ = Subset(val_dataset_, val_subset_indices)

    train_loader_ = DataLoader(
        train_dataset_,
        batch_size=batch_size_,
        shuffle=train_shuffle_,
        collate_fn=collate_
    )
    val_loader_ = DataLoader(
        val_dataset_,
        batch_size=batch_size_,
        shuffle=val_shuffle_,
        collate_fn=collate_
    )

    sinc_conv_ = SincConv(
        out_channels=128,
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

    wavlm_model_ = WavLMLarge(wavlm_path=wavlm_dir_)

    wavlm_rawnetx_sv_base = Fusion(
        wavlm_model=wavlm_model_,
        rawnetx_model=rawnetx_model_,
        wavlm_embedding_dim=wavlm_embedding_dim_,
        rawnetx_embedding_dim=rawnetx_embedding_dim_,
        fusion_embedding_dim=fusion_embedding_dim_
    ).to(device_)

    labels_in_subset = [train_dataset_[idx][1] for idx in range(len(train_dataset_))]
    num_spk_train_ = len(set(labels_in_subset))
    print("num_spk_train (subset) =", num_spk_train_)

    amsoftmax_ = AMSoftmaxLoss(
        embed_dim=fusion_embedding_dim_,
        n_classes=num_spk_train_,
        margin=amsoftmax_margin_,
        scale=amsoftmax_scale_
    ).to(device_)

    optimizer_ = torch.optim.AdamW(
        list(wavlm_rawnetx_sv_base.parameters()) + list(amsoftmax_.parameters()),
        lr=learning_rate_
    )

    metrics_ = Metrics()
    trainer_ = Trainer(
        train_loader=train_loader_,
        val_loader=val_loader_,
        model_sv=wavlm_rawnetx_sv_base,
        amsoftmax=amsoftmax_,
        optimizer=optimizer_,
        metrics=metrics_,
        device="cpu",
        epochs=epochs_,
        save_model_path=model_save_path_
    )

    trainer_.run()
