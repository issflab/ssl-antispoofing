from __future__ import annotations

from utils import create_optimizer


def build_optimizer(model, cfg, train_loader_len: int):
    optim_config = {
        "optimizer": cfg.training.optimizer,
        "amsgrad": str(cfg.training.amsgrad),
        "base_lr": cfg.training.lr,
        "lr_min": cfg.training.lr_min,
        "betas": cfg.training.betas,
        "weight_decay": cfg.training.weight_decay,
        "scheduler": cfg.training.scheduler,
        "epochs": cfg.training.epochs,
        "steps_per_epoch": train_loader_len,
        "momentum": cfg.training.momentum,
        "nesterov": cfg.training.nesterov,
        "milestones": cfg.training.milestones,
        "lr_decay": cfg.training.lr_decay,
        "T0": cfg.training.t0,
        "Tmult": cfg.training.tmult,
    }
    return create_optimizer(model.parameters(), optim_config)
