from __future__ import annotations

import os

from torch.utils.data import DataLoader

from data_utils_SSL import Multi_Dataset_train, parse_protocol


def build_protocol_paths(cfg):
    return (
        os.path.join(cfg.data.protocols_path, cfg.data.train_protocol),
        os.path.join(cfg.data.protocols_path, cfg.data.dev_protocol),
    )


def load_split_from_protocol(protocol_path: str, cfg):
    return parse_protocol(
        protocol_path,
        delimiter=cfg.data.protocol_delimiter,
        key_col=cfg.data.protocol_key_column,
        label_col=cfg.data.protocol_label_column,
        has_label=True,
    )


def build_datasets(args, cfg):
    train_proto, dev_proto = build_protocol_paths(cfg)
    d_label_trn, file_train = load_split_from_protocol(train_proto, cfg)
    d_label_dev, file_dev = load_split_from_protocol(dev_proto, cfg)

    train_set = Multi_Dataset_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=cfg.data.database_path,
        algo=args.algo,
        max_len=cfg.data.train_audio_max_samples,
    )
    dev_set = Multi_Dataset_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=cfg.data.database_path,
        algo=args.algo,
        max_len=cfg.data.eval_audio_max_samples,
    )

    return {
        "train_proto": train_proto,
        "dev_proto": dev_proto,
        "train_ids": file_train,
        "dev_ids": file_dev,
        "train_set": train_set,
        "dev_set": dev_set,
    }


def build_dataloaders(datasets, cfg):
    train_loader = DataLoader(
        datasets["train_set"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        datasets["dev_set"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
    )
    return train_loader, dev_loader
