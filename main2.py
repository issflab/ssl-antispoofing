import argparse
import os

import torch
from tensorboardX import SummaryWriter
from torchcontrib.optim import SWA

from anti_spoofing.data import build_dataloaders, build_datasets
from anti_spoofing.engine import run_eval_loop, run_training_loop
from anti_spoofing.losses import build_loss
from anti_spoofing.models import build_model
from anti_spoofing.optim import build_optimizer
from config import load_config_from_env
from core_scripts.startup_config import set_random_seed

__author__ = "Hashim Ali"
__email__ = "alhashim@umich.edu"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="SSL Anti-Spoofing entry point")
    parser.add_argument(
        "--env_file",
        type=str,
        default="example_configs/aasist_codecfake.env",
        help="Structured environment config for this experiment",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = load_config_from_env(args.env_file)
    runtime_args = cfg.to_runtime_args()

    set_random_seed(cfg.experiment.seed, runtime_args)

    print(cfg.model.backend)
    print(cfg.data.dataset_name)
    print(cfg.data.database_path)

    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    writer = SummaryWriter(cfg.log_dir)

    model = build_model(cfg.model.backend, runtime_args, device).to(device)
    print("nb_params:", sum(p.numel() for p in model.parameters()))

    checkpoint_path = cfg.evaluation.checkpoint
    if checkpoint_path:
        print("Loading pretrained checkpoint from", checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    datasets = build_datasets(runtime_args, cfg)
    print("no. of training trials", len(datasets["train_ids"]))
    print("no. of validation trials", len(datasets["dev_ids"]))

    train_loader, dev_loader = build_dataloaders(datasets, cfg)
    optimizer, scheduler = build_optimizer(model, cfg, len(train_loader))
    del scheduler

    class_weight = torch.FloatTensor(cfg.training.class_weight).to(device)
    criterion = build_loss(
        cfg.model.loss,
        device,
        class_weight=class_weight,
        focal_gamma=cfg.model.focal_gamma,
    )
    optimizer_swa = SWA(optimizer)

    if cfg.training.mode == "train":
        run_training_loop(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            optimizer_swa=optimizer_swa,
            criterion=criterion,
            train_loader=train_loader,
            dev_loader=dev_loader,
            writer=writer,
            model_save_path=cfg.checkpoint_dir,
            metric_path=cfg.metric_dir,
            dev_proto=datasets["dev_proto"],
            device=device,
        )
    elif cfg.training.mode == "eval":
        if not checkpoint_path:
            candidate = os.path.join(cfg.checkpoint_dir, "swa.pth")
            if os.path.isfile(candidate):
                print("Loading checkpoint from", candidate)
                model.load_state_dict(torch.load(candidate, map_location=device))

        run_eval_loop(
            cfg=cfg,
            model=model,
            dev_loader=dev_loader,
            criterion=criterion,
            metric_path=cfg.metric_dir,
            dev_proto=datasets["dev_proto"],
            device=device,
        )
    else:
        raise ValueError("training.mode must be 'train' or 'eval'")


if __name__ == "__main__":
    main()
