"""
config.py
---------
Configuration management for ASD_SUPERB.

This file defines the Config dataclass, which centralizes all configuration options
for model architecture, dataset selection, file paths, training/evaluation modes, and device settings.

Environment variables can override default values for flexible deployment.

Classes:
    Config: Main configuration class with properties for protocol paths and model saving.

Usage:
    Import cfg from this file to access configuration throughout the project.
    Example:
        from config import cfg
        print(cfg.model_arch)

Configuration Options:
    model_arch: Backend Model architecture to use ('aasist', 'sls', 'linear_head').
    dataset: Name of the dataset(s) used for training. Used for naming conventions.
    database_path: Root directory containing audio data.
    protocols_path: Directory containing protocol files.
    train_protocol: Filename for training protocol.
    dev_protocol: Filename for development protocol.
    mode: 'train' or 'eval' to set the running mode.
    save_dir: Directory to save models and logs.
    model_name: Name/tag for the current model run.
    cuda_device: CUDA device string (e.g., 'cuda:0').
    pretrained_checkpoint: Optional path to a pretrained model checkpoint.

Methods:
    train_protocol_path: Returns full path to training protocol file.
    dev_protocol_path: Returns full path to development protocol file.
    model_save_path: Returns directory path for saving model checkpoints.
    prepare_dirs: Creates necessary directories for saving outputs.

Environment Variables:
    SSL_MODEL_ARCH
    SSL_DATABASE_PATH
    SSL_PROTOCOLS_PATH
    SSL_MODE
    SSL_MODEL_NAME
    CUDA_DEVICE
    SSL_PRETRAINED_CHECKPOINT
"""

from dataclasses import dataclass
from typing import Optional, Literal
import os

@dataclass
class Config:
    #'aasist', 'sls', or 'xlsrmamba'
    model_arch: Literal['aasist', 'sls', 'xlsrmamba'] = 'aasist'

    # Dataset name
    dataset: str = 'Codec_FF_ITW_Pod_mlaad_spoofceleb'

    database_path: str = '/data/Data'   # root that contains e.g. spoofceleb/flac/...
    protocols_path: str = '/data/Data'  

    train_protocol: str = 'SAFE_Challenge_train_protocol_v3.txt'
    dev_protocol: str = 'SAFE_Challenge_dev_protocol_V3.txt'

    mode: Literal['train', 'eval'] = 'train'

    save_dir: str = './output/models'
    model_name: str = 'run1'

    cuda_device: str = 'cuda:1'

    pretrained_checkpoint: Optional[str] = None

    @property
    def train_protocol_path(self) -> str:
        return os.path.join(self.protocols_path, self.train_protocol)

    @property
    def dev_protocol_path(self) -> str:
        return os.path.join(self.protocols_path, self.dev_protocol)

    @property
    def model_save_path(self) -> str:
        return os.path.join(self.save_dir, self.model_name)

    def prepare_dirs(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)


cfg = Config()

cfg.model_arch = os.getenv('SSL_MODEL_ARCH', cfg.model_arch)
cfg.database_path = os.getenv('SSL_DATABASE_PATH', cfg.database_path)
cfg.protocols_path = os.getenv('SSL_PROTOCOLS_PATH', cfg.protocols_path)
cfg.mode = os.getenv('SSL_MODE', cfg.mode)
cfg.model_name = os.getenv('SSL_MODEL_NAME', cfg.model_name)
env_ckpt = os.getenv('SSL_PRETRAINED_CHECKPOINT')
if env_ckpt:
    cfg.pretrained_checkpoint = env_ckpt

cfg.prepare_dirs()
