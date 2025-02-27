# Third-party imports
from pydantic import BaseModel, conint, confloat


# CUDA
class CudaConfig(BaseModel):
    cuda_alloc_conf: str = "expandable_segments:True"


# Data
class DataTestPaths(BaseModel):
    wavs: str
    pairs: str


class DataPaths(BaseModel):
    root: str
    test: DataTestPaths


class SegmentConfig(BaseModel):
    length: confloat(gt=0.0) = 2.0


class DataConfig(BaseModel):
    paths: DataPaths
    rate: conint(gt=0) = 16000
    segment: SegmentConfig = SegmentConfig()


# RawNetX
class SincConfig(BaseModel):
    out_channels: conint(gt=0) = 128
    kernel_size: conint(gt=0) = 251
    in_channels: conint(gt=0) = 1


class RawNetXBlockConfig(BaseModel):
    num_blocks: conint(gt=0) = 4
    kernel_size: conint(gt=0) = 3
    dilation: conint(gt=0) = 1
    use_se: bool = True


class RawNetXPoolingConfig(BaseModel):
    attention_dim: conint(gt=0) = 128


class RawNetXConfig(BaseModel):
    embedding_dim: conint(gt=0) = 256
    sinc: SincConfig = SincConfig()
    block: RawNetXBlockConfig = RawNetXBlockConfig()
    pooling: RawNetXPoolingConfig = RawNetXPoolingConfig()


# WavLM Config
class WavLMConfig(BaseModel):
    path: str = "microsoft/wavlm-large"
    embedding_dim: conint(gt=0) = 1024


# Fusion
class FusionConfig(BaseModel):
    embedding_dim: conint(gt=0) = 256


# Training
class TrainingConfig(BaseModel):
    batch_size: conint(gt=0) = 4
    epochs: conint(gt=0) = 3
    learning_rate: confloat(gt=0.0) = 1e-5
    train_shuffle: bool = True
    val_shuffle: bool = False


# Loss
class AMSoftmaxConfig(BaseModel):
    scale: confloat(gt=0.0) = 30.0
    margin: confloat(ge=0.0) = 0.3


class LossConfig(BaseModel):
    amsoftmax: AMSoftmaxConfig = AMSoftmaxConfig()


# WavLMRawNetXSVBase
class WavLMRawNetXSVBasePath(BaseModel):
    save: str = ".model/WavLMRawNetXSVBase.pt"


class WavLMRawNetXSVBaseConfig(BaseModel):
    path: WavLMRawNetXSVBasePath = WavLMRawNetXSVBasePath()


# RootConfig
class RootConfig(BaseModel):
    cuda: CudaConfig = CudaConfig()
    data: DataConfig
    rawnetx: RawNetXConfig
    wavlm: WavLMConfig = WavLMConfig()
    fusion: FusionConfig = FusionConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()
    wavlmrawnetxsvbase: WavLMRawNetXSVBaseConfig = WavLMRawNetXSVBaseConfig()
