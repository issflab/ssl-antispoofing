from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from types import SimpleNamespace
from typing import Any, Optional

from dotenv import load_dotenv


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _parse_list(value: str, current: list[Any]) -> list[Any]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not current:
        return items
    sample = current[0]
    if isinstance(sample, bool):
        return [_parse_bool(item) for item in items]
    if isinstance(sample, int):
        return [int(item) for item in items]
    if isinstance(sample, float):
        return [float(item) for item in items]
    return items


def _coerce_value(raw: str, current: Any) -> Any:
    if raw == "" and current is None:
        return None
    if isinstance(current, bool):
        return _parse_bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return _parse_list(raw, current)
    if current is None:
        if raw.lower() in {"none", "null"}:
            return None
        return raw
    return raw


def _apply_nested_override(obj: Any, key_path: list[str], raw_value: str):
    target = obj
    for part in key_path[:-1]:
        target = getattr(target, part.lower())
    attr = key_path[-1].lower()
    current = getattr(target, attr)
    setattr(target, attr, _coerce_value(raw_value, current))


def _merge_dataclass(obj: Any, values: dict[str, Any]):
    for key, value in values.items():
        attr = key.lower()
        if not hasattr(obj, attr):
            continue
        current = getattr(obj, attr)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, attr, value)


@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    output_dir: str = "outputs"
    seed: int = 1234
    comment: Optional[str] = None


@dataclass
class RawBoostConfig:
    algo: int = 5
    nbands: int = 5
    minf: int = 20
    maxf: int = 8000
    minbw: int = 100
    maxbw: int = 1000
    mincoeff: int = 10
    maxcoeff: int = 100
    ming: int = 0
    maxg: int = 0
    minbiaslinnonlin: int = 5
    maxbiaslinnonlin: int = 20
    n_f: int = 5
    p: int = 10
    g_sd: int = 2
    snrmin: int = 10
    snrmax: int = 40


@dataclass
class DataConfig:
    dataset_name: str = "ITW"
    database_path: str = "/data/Data/"
    protocols_path: str = "/data/Data/ds_wild/protocols/"
    train_protocol: str = "ASVspoof2019_train_protocol.txt"
    dev_protocol: str = "ASVspoof2019_dev_protocol.txt"
    eval_protocol: Optional[str] = None
    sampling_rate: int = 16000
    train_audio_max_samples: int = 192000
    eval_audio_max_samples: int = 192000
    protocol_delimiter: Optional[str] = " "
    protocol_key_column: int = 0
    protocol_src_column: int = 1
    protocol_label_column: int = 2
    rawboost: RawBoostConfig = field(default_factory=RawBoostConfig)


@dataclass
class ModelConfig:
    frontend: str = "wavlm_large"
    frontend_layers: int = 24
    freeze_ssl: bool = True
    frontend_feature_mode: str = "featurizer"
    frontend_layer_index: int = -1
    backend: str = "aasist"
    backend_config: Optional[str] = None
    backend_settings: dict[str, Any] = field(default_factory=dict)
    loss: str = "weighted_cce"
    focal_gamma: float = 2.0
    emb_size: int = 256
    num_encoders: int = 12


@dataclass
class TrainingConfig:
    mode: str = "train"
    device: str = "cuda:0"
    batch_size: int = 14
    epochs: int = 50
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    lr_min: float = 5e-6
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    amsgrad: bool = False
    momentum: float = 0.9
    nesterov: bool = False
    milestones: list[int] = field(default_factory=list)
    lr_decay: float = 0.1
    t0: int = 10
    tmult: int = 2
    metric: str = "EER"
    swa: bool = True
    num_workers: int = 8
    class_weight: list[float] = field(default_factory=lambda: [0.1, 0.9])
    cudnn_deterministic_toggle: bool = True
    cudnn_benchmark_toggle: bool = False


@dataclass
class EvaluationConfig:
    checkpoint: Optional[str] = None
    save_score_file: bool = True
    score_file_name: str = "eval_score.txt"


@dataclass
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.experiment.output_dir, self.experiment.name)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.run_dir, "logs")

    @property
    def metric_dir(self) -> str:
        return os.path.join(self.run_dir, "metrics")

    def prepare_dirs(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.metric_dir, exist_ok=True)

    def to_runtime_args(self) -> SimpleNamespace:
        rb = self.data.rawboost
        return SimpleNamespace(
            ssl_model=self.model.frontend,
            freeze_ssl=self.model.freeze_ssl,
            frontend_feature_mode=self.model.frontend_feature_mode,
            frontend_layer_index=self.model.frontend_layer_index,
            emb_size=self.model.emb_size,
            num_encoders=self.model.num_encoders,
            backend_config=self.model.backend_settings,
            frontend_layers=self.model.frontend_layers,
            algo=rb.algo,
            nBands=rb.nbands,
            minF=rb.minf,
            maxF=rb.maxf,
            minBW=rb.minbw,
            maxBW=rb.maxbw,
            minCoeff=rb.mincoeff,
            maxCoeff=rb.maxcoeff,
            minG=rb.ming,
            maxG=rb.maxg,
            minBiasLinNonLin=rb.minbiaslinnonlin,
            maxBiasLinNonLin=rb.maxbiaslinnonlin,
            N_f=rb.n_f,
            P=rb.p,
            g_sd=rb.g_sd,
            SNRmin=rb.snrmin,
            SNRmax=rb.snrmax,
            cudnn_deterministic_toggle=self.training.cudnn_deterministic_toggle,
            cudnn_benchmark_toggle=self.training.cudnn_benchmark_toggle,
        )


def _load_model_defaults(cfg: AppConfig, model_config_path: str):
    if not model_config_path:
        return
    with open(model_config_path, "r") as handle:
        model_defaults = json.load(handle)

    for section_name in ("model", "training", "evaluation"):
        section_values = model_defaults.get(section_name)
        if section_values:
            _merge_dataclass(getattr(cfg, section_name), section_values)


def load_config_from_env(env_file: Optional[str] = None) -> AppConfig:
    if env_file:
        load_dotenv(env_file, override=True)

    cfg = AppConfig()

    model_config_path = os.getenv("MODEL__BACKEND_CONFIG", cfg.model.backend_config or "")
    if model_config_path:
        cfg.model.backend_config = model_config_path
        _load_model_defaults(cfg, model_config_path)

    for key, value in os.environ.items():
        if "__" not in key:
            continue
        parts = key.split("__")
        top_level = parts[0].lower()
        if not hasattr(cfg, top_level):
            continue
        _apply_nested_override(cfg, parts, value)

    cfg.prepare_dirs()
    return cfg
