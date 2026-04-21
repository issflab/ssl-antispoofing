from __future__ import annotations

import os

import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from data_utils_SSL import _normalize_delim
from evaluation import calculate_EER


def evaluate_accuracy(dev_loader, model, device, criterion):
    model.eval()
    val_loss = 0.0
    num_total = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_x, utt_id, batch_y in dev_loader:
            del utt_id
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size

            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel().tolist()
            pred = ["fake" if bs < 0 else "bonafide" for bs in batch_score]
            keys = ["fake" if by == 0 else "bonafide" for by in batch_y.tolist()]
            y_pred.extend(pred)
            y_true.extend(keys)

    avg_loss = val_loss / num_total if num_total > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred) if y_true else 0.0
    return avg_loss, balanced_acc


def produce_evaluation(
    data_loader,
    model,
    device,
    criterion,
    save_path,
    trial_path,
    trial_delimiter=None,
    trial_cols_utt=0,
    trial_cols_src=3,
    trial_cols_label=4,
):
    model.eval()
    val_loss = 0.0
    num_total = 0.0

    fname_list = []
    score_list = []

    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()

    with torch.no_grad():
        for batch_x, utt_id, batch_y in tqdm(data_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel().tolist()
            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size

            fname_list.extend(utt_id)
            score_list.extend(batch_score)

    assert len(trial_lines) == len(fname_list) == len(score_list)
    tdelim = _normalize_delim(trial_delimiter)

    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            parts = trl.strip().split(tdelim) if tdelim is not None else trl.strip().split()
            try:
                utt_id, src, key = (
                    parts[trial_cols_utt],
                    parts[trial_cols_src],
                    parts[trial_cols_label],
                )
            except IndexError as exc:
                raise ValueError(
                    f"Trial line has too few columns for configured indices: `{trl.strip()}`"
                ) from exc

            assert fn == utt_id
            fh.write(f"{utt_id} {src} {key} {sco}\n")

    avg_loss = val_loss / num_total if num_total > 0 else 0.0
    print(f"Scores saved to {save_path}")
    return avg_loss


def train_epoch(train_loader, model, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    num_total = 0.0

    for batch_x, utt_id, batch_y in tqdm(train_loader):
        del utt_id
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return running_loss / num_total if num_total > 0 else 0.0


def run_training_loop(
    cfg,
    model,
    optimizer,
    optimizer_swa,
    criterion,
    train_loader,
    dev_loader,
    writer,
    model_save_path,
    metric_path,
    dev_proto,
    device,
):
    metric_name = cfg.training.metric.lower()
    best_metric = float("inf") if metric_name == "eer" else float("-inf")
    n_swa_update = 0

    for epoch in range(cfg.training.epochs):
        train_loss = train_epoch(train_loader, model, optimizer, device, criterion)

        if metric_name == "eer":
            val_loss = produce_evaluation(
                dev_loader,
                model,
                device,
                criterion,
                os.path.join(metric_path, "dev_score.txt"),
                dev_proto,
                trial_delimiter=cfg.data.protocol_delimiter,
                trial_cols_utt=cfg.data.protocol_key_column,
                trial_cols_src=cfg.data.protocol_src_column,
                trial_cols_label=cfg.data.protocol_label_column,
            )
            metric_value = calculate_EER(cm_scores_file=os.path.join(metric_path, "dev_score.txt"))
            is_improved = metric_value < best_metric
        else:
            val_loss, metric_value = evaluate_accuracy(dev_loader, model, device, criterion)
            is_improved = metric_value > best_metric

        metric_tag = cfg.training.metric.lower()
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar(f"val_{metric_tag}", metric_value, epoch)
        print(
            f"Epoch {epoch} - train_loss: {train_loss:.4f} - "
            f"val_loss: {val_loss:.4f} - val_{metric_tag}: {metric_value:.4f}"
        )

        if is_improved:
            print(f"Best model updated at epoch {epoch}")
            best_metric = metric_value
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, f"epoch_{epoch}_{metric_value:03.3f}.pth"),
            )
            optimizer_swa.update_swa()
            n_swa_update += 1

        writer.add_scalar(f"best_val_{metric_tag}", best_metric, epoch)

    print("Finalizing SWA (if any updates occurred)")
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(train_loader, model, device=device)
        torch.save(model.state_dict(), os.path.join(model_save_path, "swa.pth"))


def run_eval_loop(cfg, model, dev_loader, criterion, metric_path, dev_proto, device):
    val_loss, val_balanced_acc = evaluate_accuracy(dev_loader, model, device, criterion)
    print(f"EVAL: val_loss={val_loss:.4f}, balanced_acc={val_balanced_acc:.4f}")

    if cfg.evaluation.save_score_file:
        eval_score_path = os.path.join(metric_path, cfg.evaluation.score_file_name)
        produce_evaluation(
            dev_loader,
            model,
            device,
            criterion,
            eval_score_path,
            dev_proto,
            trial_delimiter=cfg.data.protocol_delimiter,
            trial_cols_utt=cfg.data.protocol_key_column,
            trial_cols_src=cfg.data.protocol_src_column,
            trial_cols_label=cfg.data.protocol_label_column,
        )
        eval_eer = calculate_EER(cm_scores_file=eval_score_path)
        print("EVAL: score_file =", eval_score_path)
        print(f"EVAL: eer={eval_eer:.4f}")
