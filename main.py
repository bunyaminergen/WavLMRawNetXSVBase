# Standard library imports
import os
import time

# Third-party imports
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Local imports
from src.process.test import Tester
from src.model.fusion import Fusion
from src.process.train import Trainer
from src.evaluate.metric import Metrics
from src.model.loss import AMSoftmaxLoss
from src.config.schema import RootConfig
from src.model.block import ResidualStack
from src.model.convolution import SincConv
from src.preprocess.feature import Segment
from src.model.backbone import RawNetX, WavLMLarge
from src.model.pooling import AttentiveStatsPooling
from src.preprocess.transformation import Transform
from src.utils.data.manager import VoxCeleb1Dataset, Collate


def main() -> None:
    """
    Main entry point for the training and testing of a speaker verification model
    based on a fusion of WavLM and RawNetX.

    This function:
      1. Loads a YAML configuration using `omegaconf`.
      2. Sets up CUDA environment variables.
      3. Initializes datasets, DataLoaders, and any required data preprocessing.
      4. Constructs models (RawNetX, WavLM) and fuses them with a `Fusion` module.
      5. Creates and trains a Trainer with an AMSoftmax loss, then saves the trained model.
      6. Loads the best model checkpoint for final testing with a `Tester`.
      7. Reports total execution time.

    The function expects that paths, parameters, and hyperparameters are
    defined in a configuration file pointed to by `config_path`. Adjust the
    script according to your own project structure if needed.

    Examples
    --------
    Suppose you have a `config.yaml` file properly set up under `src/config/config.yaml`.
    You can run this script from the command line:

    >>> # python main.py  # doctest: +SKIP

    After running, the script will:
      - Train the model on the specified dataset.
      - Evaluate the model on a validation set.
      - Save the best model checkpoint.
      - Finally, run an additional test procedure and report the EER and minDCF.

    Raises
    ------
    FileNotFoundError
        If the specified config file or dataset paths are not found.
    ValueError
        If any configuration parameter is incorrect (e.g., negative batch size).
    """

    start_time = time.time()

    # Configuration
    ## Config File Load
    config_path = "src/config/config.yaml"
    config_load = OmegaConf.load(config_path)
    conf_dict = OmegaConf.to_container(config_load, resolve=True)
    config = RootConfig(**conf_dict)

    ## CUDA
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.cuda.cuda_alloc_conf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Data
    ### Paths
    dataset_root_path = config.data.paths.root
    test_wav_root = config.data.paths.test.wavs
    test_pairs_txt = config.data.paths.test.pairs
    ### Preprocess
    sample_rate = config.data.rate
    segment_length = config.data.segment.length

    ## RawNetX
    rawnetx_embedding_dim = config.rawnetx.embedding_dim
    ### Sinc Convolution
    sinc_out_channels = config.rawnetx.sinc.out_channels
    sinc_kernel_size = config.rawnetx.sinc.kernel_size
    sinc_in_channels = config.rawnetx.sinc.in_channels
    ### Residual Blocks
    num_blocks = config.rawnetx.block.num_blocks
    block_kernel_size = config.rawnetx.block.kernel_size
    block_dilation = config.rawnetx.block.dilation
    block_use_se = config.rawnetx.block.use_se
    ### Attentive Stats Pooling
    pooling_attention_dim = config.rawnetx.pooling.attention_dim

    ## WavLM Large
    wavlm_path = config.wavlm.path
    wavlm_embedding_dim = config.wavlm.embedding_dim

    ## Fusion
    fusion_embedding_dim = config.fusion.embedding_dim

    ## WavLM RawNetX SV Base
    model_save_path = config.wavlmrawnetxsvbase.path.save

    ## Training
    batch_size = config.training.batch_size
    epochs = config.training.epochs
    learning_rate = config.training.learning_rate
    train_shuffle = config.training.train_shuffle
    val_shuffle = config.training.val_shuffle

    ## Loss
    amsoftmax_margin = config.loss.amsoftmax.margin
    amsoftmax_scale = config.loss.amsoftmax.scale

    # Initial Classes
    transform = Transform()
    collate = Collate()
    metrics = Metrics()

    # Dataset Preparation & Loading
    train_dataset = VoxCeleb1Dataset(
        dataset_root=dataset_root_path,
        segment_length=segment_length,
        sample_rate=sample_rate,
        transform=transform,
        segment_class=Segment
    )
    val_dataset = VoxCeleb1Dataset(
        dataset_root=dataset_root_path,
        subset='valid',
        segment_length=segment_length,
        sample_rate=sample_rate,
        transform=transform,
        segment_class=Segment
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=val_shuffle,
        collate_fn=collate
    )

    num_speaker_train = len(train_dataset.speaker_map)

    # RawNetX
    ## Sinc Convolution
    sinc_conv = SincConv(
        out_channels=sinc_out_channels,
        kernel_size=sinc_kernel_size,
        sample_rate=sample_rate,
        in_channels=sinc_in_channels,
        padding=sinc_kernel_size // 2
    )

    ## Residual Blocks
    residual_stack = ResidualStack(
        channels=sinc_out_channels,
        kernel_size=block_kernel_size,
        dilation=block_dilation,
        use_se=block_use_se,
        num_blocks=num_blocks
    )

    ## Attentive Stats Pooling
    pooling_layer = AttentiveStatsPooling(
        in_dim=sinc_out_channels,
        attention_dim=pooling_attention_dim
    )

    ## RawnetX Model
    rawnetx_model = RawNetX(
        sinc_conv=sinc_conv,
        residual_blocks=residual_stack,
        pooling_layer=pooling_layer,
        sinc_out_channels=sinc_out_channels,
        rawnetx_embedding_dim=rawnetx_embedding_dim
    )

    # WavLM Large
    wavlm_model = WavLMLarge(wavlm_path=wavlm_path)

    # Fusion Layer
    wavlm_rawnetx_sv_base = Fusion(
        wavlm_model=wavlm_model,
        rawnetx_model=rawnetx_model,
        wavlm_embedding_dim=wavlm_embedding_dim,
        rawnetx_embedding_dim=rawnetx_embedding_dim,
        fusion_embedding_dim=fusion_embedding_dim
    ).to(device)

    # Loss & Optimizer
    amsoftmax = AMSoftmaxLoss(
        embed_dim=fusion_embedding_dim,
        n_classes=num_speaker_train,
        margin=amsoftmax_margin,
        scale=amsoftmax_scale
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(wavlm_rawnetx_sv_base.parameters()) + list(amsoftmax.parameters()),
        lr=learning_rate
    )

    # Training
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model_sv=wavlm_rawnetx_sv_base,
        amsoftmax=amsoftmax,
        optimizer=optimizer,
        metrics=metrics,
        epochs=epochs,
        save_model_path=model_save_path
    )
    trainer.run()

    # Test
    checkpoint = torch.load(model_save_path, map_location=device)
    wavlm_rawnetx_sv_base.load_state_dict(checkpoint["model_state"])
    wavlm_rawnetx_sv_base.eval()

    tester = Tester(
        model_sv=wavlm_rawnetx_sv_base,
        transform=transform,
        metrics=metrics,
        test_wav_root=test_wav_root,
        test_pairs_txt=test_pairs_txt,
        sample_rate=sample_rate
    )
    tester.run()

    end_time = time.time()
    print(f"End-to-end process completed. Total time: {(end_time - start_time):.3f} sec | "
          f"{(end_time - start_time) / 60:.3f} min | {(end_time - start_time) / 3600:.3f} hours")


if __name__ == "__main__":
    main()
