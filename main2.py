import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from tensorboardX import SummaryWriter
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import json
from s3prl import hub
from data_utils_SSL import genSpoof_list_multidata, Multi_Dataset_train, parse_protocol, _normalize_delim
from aasist_model import Model as aasist_model
from sls_model import Model as sls_model
# from xlsrmamba_model import Model as XLSRMambaModel
from core_scripts.startup_config import set_random_seed
from config import cfg
from utils import create_optimizer
from evaluation import calculate_EER
import time

# --- NEW: import the generic, config-driven protocol parser
__author__ = "Hashim Ali"
__email__ = "alhashim@umich.edu"


def evaluate_accuracy(dev_loader, model, device):
    model.eval()
    val_loss = 0.0
    num_total = 0.0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_x, utt_id, batch_y in dev_loader:
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
    balanced_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    return avg_loss, balanced_acc


# --- UPDATED: make trial parsing config-driven while keeping behavior the same by default
def produce_evaluation(data_loader, model, device, save_path, trial_path,
                       # NEW: pass trial schema from config
                       trial_delimiter=None, trial_cols_utt=0, trial_cols_src=3, trial_cols_label=4):
    model.eval()

    val_loss = 0.0
    num_total = 0.0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    fname_list = []
    key_list = []
    score_list = []

    trial_lines = []

    with open(trial_path, "r") as f_trl:
        trial_lines.extend(f_trl.readlines())
    
    with torch.no_grad():
        for batch_x, utt_id, batch_y in tqdm(data_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() 
            batch_score = batch_score.tolist()

            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
            # Writing score to file
            fname_list.extend(utt_id)
            score_list.extend(batch_score)
            key_list.extend(batch_y)
    
    # print(len(fname_list), len(key_list), len(score_list), len(trial_lines))
    assert len(trial_lines) == len(fname_list) == len(score_list)

    tdelim = _normalize_delim(trial_delimiter)

    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            parts = trl.strip().split(tdelim) if tdelim is not None else trl.strip().split()
            # --- UPDATED: read configured columns
            try:
                utt_id, src, key = parts[trial_cols_utt], parts[trial_cols_src], parts[trial_cols_label]
            except IndexError:
                raise ValueError(f"Trial line has too few columns for configured indices: `{trl.strip()}`")

            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))

    
    val_loss /= num_total

    print('Scores saved to {}'.format(save_path))

    return val_loss


def train_epoch(train_loader, model, optimizer, device):
    model.train()
    running_loss = 0.0
    num_total = 0.0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, utt_id, batch_y in tqdm(train_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # start = time.time()
        batch_out = model(batch_x)
        # end = time.time()

        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    avg_loss = running_loss / num_total if num_total > 0 else 0.0
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refactored SSL-AntiSpoofing entry point using cfg')
    parser.add_argument('--database_path', type=str, default=None)
    parser.add_argument('--protocols_path', type=str, default=None)
    parser.add_argument('--ssl_model', type=str, default='wavlm_large')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None, help='Pretrained checkpoint to load')
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', default=True,
                    help='use cudnn-deterministic? (default true)')
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', default=False,
                    help='use cudnn-benchmark? (default false)')
    parser.add_argument('--emb_size', type=int, default=256, help='Size of projection layer for XLSR Mamba')
    parser.add_argument('--num_encoders', type=int, default=12, help='Number of encoder layers in Mamba')
    parser.add_argument('--nBands', type=int, default=5, help='number of notch filters')
    parser.add_argument('--minF', type=int, default=20, help='minimum centre frequency [Hz]')
    parser.add_argument('--maxF', type=int, default=8000, help='maximum centre frequency [Hz]')
    parser.add_argument('--minBW', type=int, default=100, help='minimum filter width [Hz]')
    parser.add_argument('--maxBW', type=int, default=1000, help='maximum filter width [Hz]')
    parser.add_argument('--minCoeff', type=int, default=10, help='minimum filter coefficients')
    parser.add_argument('--maxCoeff', type=int, default=100, help='maximum filter coefficients')
    parser.add_argument('--minG', type=int, default=0, help='minimum gain factor of linear component')
    parser.add_argument('--maxG', type=int, default=0, help='maximum gain factor of linear component')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, help='minimum gain difference between linear and non-linear components')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, help='maximum gain difference between linear and non-linear components')
    parser.add_argument('--N_f', type=int, default=5, help='order of the (non-)linearity where N_f=1 refers only to linear components')
    parser.add_argument('--P', type=int, default=10, help='Maximum number of uniformly distributed samples in [%]')
    parser.add_argument('--g_sd', type=int, default=2, help='gain parameter > 0')
    parser.add_argument('--SNRmin', type=int, default=10, help='Minimum SNR value for coloured additive noise')
    parser.add_argument('--SNRmax', type=int, default=40, help='Maximum SNR value for coloured additive noise')
    UPSTREAM_CHOICES = [attr for attr in dir(hub) if not attr.startswith("_")]
    parser.add_argument("--h", type=str, choices=UPSTREAM_CHOICES, default=UPSTREAM_CHOICES[0], help=("Which upstream to load. Valid options are:\n  " + "\n  ".join(UPSTREAM_CHOICES)))



    # Rawboost / augmentation
    parser.add_argument('--algo', type=int, default=5)
    # (Other augmentation args can remain as before if still used)

    args = parser.parse_args()

    # Override config with CLI if provided
    if args.database_path:
        cfg.database_path = args.database_path
    if args.protocols_path:
        cfg.protocols_path = args.protocols_path
    if args.model_path:
        cfg.pretrained_checkpoint = args.model_path

    # reproducibility
    set_random_seed(args.seed, args)

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}_{}'.format(args.loss, args.num_epochs, args.batch_size, cfg.model_arch, cfg.dataset, args.ssl_model)
    
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    
    writer = SummaryWriter(f'logs/{model_tag}')

    # prepare save path
    model_save_path = os.path.join(cfg.save_dir, model_tag)
    os.makedirs(model_save_path, exist_ok=True)

    #GPU device
    device = cfg.cuda_device if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    # instantiate model based on config
    if cfg.model_arch == 'aasist':
        model = aasist_model(args, device)
    elif cfg.model_arch == 'sls':
        model = sls_model(args, device)
    elif cfg.model_arch == 'xlsrmamba':
        model = XLSRMambaModel(args, device)
    else:
        raise ValueError(f'Unknown model architecture: {cfg.model_arch}')
    model = model.to(device)

    nb_params = sum(p.numel() for p in model.parameters())
    print('nb_params:', nb_params)

    # load pretrained checkpoint if given
    if cfg.pretrained_checkpoint:
        print('Loading pretrained checkpoint from', cfg.pretrained_checkpoint)
        model.load_state_dict(torch.load(cfg.pretrained_checkpoint, map_location=device))

    # dataset prep
    train_proto = os.path.join(cfg.protocols_path, cfg.train_protocol)
    dev_proto = os.path.join(cfg.protocols_path, cfg.dev_protocol)

    # --- UPDATED: use config-driven parser for BOTH train and dev
    d_label_trn, file_train = parse_protocol(
        train_proto,
        delimiter=cfg.protocol_delimiter,
        key_col=cfg.protocol_key_column,
        label_col=cfg.protocol_label_column,
        has_label=True
    )
    d_label_dev, file_dev = parse_protocol(
        dev_proto,
        delimiter=cfg.protocol_delimiter,
        key_col=cfg.protocol_key_column,
        label_col=cfg.protocol_label_column,
        has_label=True
    )
    # ----------------------------------------------------------------

    # print("Train protocol:", getattr(cfg, "train_protocol_path", cfg.train_protocol))
    # print("Dev protocol:", getattr(cfg, "dev_protocol_path", cfg.dev_protocol))
    # print("Database base path:", cfg.database_path)
    # for tag, file_list in (("TRAIN", file_train), ("DEV", file_dev)):
    #     print(f"Sample {tag} utt_ids (first 3):", file_list[:3])
    #     for utt in file_list[:3]:
    #         full = os.path.join(cfg.database_path, utt)
    #         print(f"  [{tag}] {utt} -> {full} exists: {os.path.isfile(full)}")

    train_set = Multi_Dataset_train(args, list_IDs=file_train, labels=d_label_trn,
                                    base_dir=cfg.database_path, algo=args.algo)
    dev_set = Multi_Dataset_train(args, list_IDs=file_dev, labels=d_label_dev,
                                  base_dir=cfg.database_path, algo=args.algo)

    print('no. of training trials', len(file_train))
    print('no. of validation trials', len(file_dev))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=8, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,
                            num_workers=8, shuffle=False)

    # optimizer + scheduler
    # load external config if exists

    # instantiate model based on config
    if cfg.model_arch == 'aasist':
        with open("./AASIST.conf", "r") as f_json:
            args_config = json.loads(f_json.read())
        optim_config = args_config["optim_config"]
        optim_config["epochs"] = args.num_epochs
        optim_config["steps_per_epoch"] = len(train_loader)
        optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    
    elif cfg.model_arch == 'sls':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    elif cfg.model_arch == 'xlsrmamba':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    optimizer_swa = SWA(optimizer)


    # make directory for metric logging
    metric_path = os.path.join(model_tag, "metrics")
    os.makedirs(metric_path, exist_ok=True)

    os.path.join(metric_path, "dev_score.txt")
    # train vs eval
    if cfg.mode == 'train':

        best_val_eer = 10
        n_swa_update = 0

        for epoch in range(args.num_epochs):
            train_loss = train_epoch(train_loader, model, optimizer, device)

            # --- UPDATED: pass trial schema from cfg
            val_loss = produce_evaluation(
                dev_loader, model, device, os.path.join(metric_path, "dev_score.txt"), dev_proto,
                trial_delimiter=cfg.protocol_delimiter,
                trial_cols_utt=cfg.protocol_key_column,
                trial_cols_src=cfg.protocol_src_column,
                trial_cols_label=cfg.protocol_label_column
            )

            dev_eer = calculate_EER(cm_scores_file=os.path.join(metric_path, "dev_score.txt"))

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_eer', dev_eer, epoch)
            print(f"Epoch {epoch} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_eer: {dev_eer:.4f}")

            if dev_eer < best_val_eer:
                print(f"Best model updated at epoch {epoch}")
                best_val_eer = dev_eer
                torch.save(model.state_dict(),
                           os.path.join(model_save_path, "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer)))
                
                # SWA update on improvement (as per prior logic)
                print("Saving epoch {} for swa".format(epoch))
                optimizer_swa.update_swa()
                n_swa_update += 1

            writer.add_scalar("best_val_eer", best_val_eer, epoch)

        # best_bal_acc = 0.0
        # n_swa_update = 0

        # for epoch in range(args.num_epochs):
        #     train_loss = train_epoch(train_loader, model, optimizer, device)
        #     val_loss, val_balanced_acc = evaluate_accuracy(dev_loader, model, device)

        #     writer.add_scalar('train_loss', train_loss, epoch)
        #     writer.add_scalar('val_loss', val_loss, epoch)
        #     writer.add_scalar('val_balanced_acc', val_balanced_acc, epoch)
        #     print(f"Epoch {epoch} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_balanced_acc: {val_balanced_acc:.4f}")

        #     if val_balanced_acc >= best_bal_acc:
        #         print(f"Best model updated at epoch {epoch}")
        #         best_bal_acc = val_balanced_acc
        #         torch.save(model.state_dict(),
        #                    os.path.join(model_save_path, f"epoch_{epoch}_{val_balanced_acc:0.3f}.pth"))
        #         torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
        #         # SWA update on improvement (as per prior logic)
        #         optimizer_swa.update_swa()
        #         n_swa_update += 1

        #     writer.add_scalar("best_val_balanced_acc", best_bal_acc, epoch)

        print("Finalizing SWA (if any updates occurred)")
        if n_swa_update > 0:
            optimizer_swa.swap_swa_sgd()
            optimizer_swa.bn_update(train_loader, model, device=device)
            torch.save(model.state_dict(), os.path.join(model_save_path, "swa.pth"))

    elif cfg.mode == 'eval':
        # fallback to best.pth if no explicit checkpoint
        if not cfg.pretrained_checkpoint:
            candidate = os.path.join(cfg.save_dir, cfg.model_name, 'best.pth')
            if os.path.isfile(candidate):
                print('Loading best checkpoint from', candidate)
                model.load_state_dict(torch.load(candidate, map_location=device))
        val_loss, val_balanced_acc = evaluate_accuracy(dev_loader, model, device)
        print(f'EVAL: val_loss={val_loss:.4f}, balanced_acc={val_balanced_acc:.4f}')
    else:
        raise ValueError("cfg.mode must be 'train' or 'eval'")
