from dataclasses import dataclass
from typing import Optional
try:
    from typing import Literal
except ImportError:
    Literal = None

@dataclass
class DatasetConfig:
    type: Literal['ASVspoof2019', 'ASVspoof2021', 'InTheWild'] = 'ASVspoof2021'
    track: Literal['LA', 'PA', 'DF'] = 'LA'
    database_path: str = '/path/to/data'
    protocols_path: str = '/path/to/protocols'
    sample_rate: int = 16000
    max_length: int = 64600

@dataclass
class RawBoostConfig:
    algo: int = 0
    nBands: int = 5
    minF: int = 20
    maxF: int = 8000
    minBW: int = 100
    maxBW: int = 1000
    minCoeff: int = 10
    maxCoeff: int = 100
    minG: int = -20
    maxG: int = 5
    minBiasLinNonLin: int = 5
    maxBiasLinNonLin: int = 20
    N_f: int = 2
    P: int = 90
    g_sd: float = 0.2
    SNRmin: int = 10
    SNRmax: int = 30

@dataclass
class FeatureConfig:
    use_ssl: bool = True
    pretrained_model_path: str = '/path/to/xlsr2_300m.pt'
    ssl_output_dim: int = 1024

@dataclass
class TrainConfig:
    batch_size: int = 14
    num_epochs: int = 100
    lr: float = 1e-6
    weight_decay: float = 1e-4
    loss: str = 'weighted_CCE'
    seed: int = 1234
    model_path: Optional[str] = None
    comment: Optional[str] = None
    eval_output: Optional[str] = 'output/score.txt'
    eval: bool = False
    is_eval: bool = False
    eval_part: int = 0
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()
    rawboost: RawBoostConfig = RawBoostConfig()
    features: FeatureConfig = FeatureConfig()
    train: TrainConfig = TrainConfig()

cfg = Config()
