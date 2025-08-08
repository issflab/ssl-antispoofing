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
