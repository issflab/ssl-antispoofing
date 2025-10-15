"""
config.py
---------
Configuration management for spoof_SUPERB.

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
    model_arch: Literal['aasist', 'sls', 'xlsrmamba'] = 'sls'

    # Dataset name
    dataset: str = 'ASV19'

    database_path: str = '/data/Data/'   # root that contains e.g. spoofceleb/flac/...
    protocols_path: str = '/data/Data/protocols/'  

    train_protocol: str = 'ASVspoof2019_train_protocol.txt'
    dev_protocol: str = 'ASVspoof2019_dev_protocol.txt'

    mode: Literal['train', 'eval'] = 'train'

    save_dir: str = '/data/ssl_anti_spoofing/models_test'
    model_name: str = 'sls_ASV19'

    cuda_device: str = 'cuda:2'

    pretrained_checkpoint: Optional[str] = None


    # Use " " for whitespace-separated
    # "," 
    # decide these based on the protocol file format.
    protocol_delimiter: Optional[str] = " "
    protocol_key_column: int = 0
    protocol_src_column: int = 3
    protocol_label_column: int = 4

    # we don't need trial file.
    # trial_delimiter: Optional[str] = " "
    # trial_cols_utt: int = 0
    # trial_cols_src: int = 1
    # trial_cols_label: int = 2

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
cfg.train_protocol = os.getenv('SSL_TRAIN_PROTOCOL', cfg.train_protocol)
cfg.dev_protocol = os.getenv('SSL_DEV_PROTOCOL', cfg.dev_protocol)
cfg.cuda_device = os.getenv('CUDA_DEVICE', cfg.cuda_device)
cfg.mode = os.getenv('SSL_MODE', cfg.mode)
cfg.model_name = os.getenv('SSL_MODEL_NAME', cfg.model_name)
env_ckpt = os.getenv('SSL_PRETRAINED_CHECKPOINT')
if env_ckpt:
    cfg.pretrained_checkpoint = env_ckpt

# --- NEW: env overrides for protocol/trial schema ---
cfg.protocol_delimiter    = os.getenv('SSL_PROTOCOL_DELIMITER', cfg.protocol_delimiter)
cfg.protocol_key_column   = int(os.getenv('SSL_PROTOCOL_KEY_COL',  cfg.protocol_key_column))
cfg.protocol_label_column = int(os.getenv('SSL_PROTOCOL_SRC_COL', cfg.protocol_src_column))
cfg.protocol_label_column = int(os.getenv('SSL_PROTOCOL_LABEL_COL', cfg.protocol_label_column))

# cfg.trial_delimiter  = os.getenv('SSL_TRIAL_DELIMITER', cfg.trial_delimiter)
# cfg.trial_cols_utt   = int(os.getenv('SSL_TRIAL_UTT_COL',   cfg.trial_cols_utt))
# cfg.trial_cols_src   = int(os.getenv('SSL_TRIAL_SRC_COL',   cfg.trial_cols_src))
# cfg.trial_cols_label = int(os.getenv('SSL_TRIAL_LABEL_COL', cfg.trial_cols_label))
# ----------------------------------------------------

cfg.prepare_dirs()
