import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
import yaml
from data_utils_SSL import genSpoof_list_multidata, Multi_Dataset_train
from aasist_model import Model as aasist_model
from sls_model import Model as sls_model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from config import cfg
from sklearn.metrics import balanced_accuracy_score
import json
from tqdm import tqdm
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from xlsrmamba_model import Model as XLSRMambaModel


__author__ = "Hashim Ali"
__email__ = "alhashim@umich.edu"


def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    y_true = []
    y_pred = []

    for batch_x, utt_id, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        batch_score = batch_score.tolist()
        
        pred = ["fake" if bs < 0 else "bonafide" for bs in batch_score]
        keys = ["fake" if by == 0 else "bonafide" for by in batch_y.tolist()]
        y_pred.extend(pred)
        y_true.extend(keys)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
   
    return val_loss, balanced_acc


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x, utt_id, batch_y in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        batch_out = model(batch_x)
        
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() 
        batch_score = batch_score.tolist()
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score)
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))


def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, utt_id, batch_y in tqdm(train_loader):
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSL + AASIST System trained on multiple datasets')

    parser.add_argument('--database_path', type=str, default='/data/Data/', help='Change this to the base data directory which contain multiple datasets.')
    parser.add_argument('--protocols_path', type=str, default='/data/Data/protocols/', help='Change this to the path which contain protocol files')
    parser.add_argument('--ssl_feature', type=str, default='wavlm_large', help='Change this to the path which contain protocol files')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)')
    

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')

    parser.add_argument('--emb_size', type=int, default=256, help='Size of projection layer')
    parser.add_argument('--num_encoders', type=int, default=12, help='Number of encoder layers in mamba')

    
    ##===================================================Rawboost data augmentation ======================================================================#

    model_out_dir = './'   # directory to save the models in
    
    # name this variable based on datasets being used to train the models
    # CodecFake = codec
    # Famous Figures = FF
    # ASVspoof 2019 = ASV19
    # ASVspoof 2025 = ASV5
    # FakeXpose = FX
    # In the Wild = ITW
    # DFADD = DFADD
    # MLAAD = MLAAD
    # SpoofCeleb = SpoofCeleb
    # example, data_name = 'codec_FF_ASV19_MLAAD'
    # data_name = 'ASV19_CodecTTS_FF_MLAAD'
    # data_name = 'mlaad_spoofceleb_FF'
    data_name = 'Codec_FF_ITW_Pod_mlaad_spoofceleb'

    if not os.path.exists(model_out_dir):
        os.mkdir(model_out_dir)

    # train_protocol_filename = 'SAFE_challenge_train_latest_protocol.txt'
    # dev_protocol_filename = 'SAFE_challenge_dev_latest_protocol.txt'

    # train_protocol_filename = 'SAFE_challenge_train_latest_protocol.txt'
    # dev_protocol_filename = 'SAFE_challenge_dev_latest_protocol.txt'

    train_protocol_filename = 'SAFE_Challenge_train_protocol_Codec_FF_ITW_Pod_mlaad_spoofceleb.txt'
    dev_protocol_filename = 'SAFE_Challenge_dev_protocol_Codec_FF_ITW_Pod_mlaad_spoofceleb.txt'
    
    args = parser.parse_args()

    with open("./AASIST.conf", "r") as f_json:
        args_config = json.loads(f_json.read())

    optim_config = args_config["optim_config"]
    optim_config["epochs"] = args.num_epochs

    #make experiment reproducible
    set_random_seed(args.seed, args)

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}_{}'.format(args.loss, args.num_epochs, args.batch_size, args.lr, data_name, args.ssl_feature)
    
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    
    model_save_path = os.path.join(model_out_dir, model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    #GPU device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    # aasist model
    #model = aasist_model(args, device)

    # sls model
    # model = sls_model(args, device)

    # XLSR mamba model
    model = XLSRMambaModel(args, device)


    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))


    #evaluation 
    # if args.eval:
    #     file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'/ASVspoof2019.LA.cm.train.trn.txt'.format(prefix,prefix_2021)),is_train=False,is_eval=True)
    #     #file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'fakeXpose_protocol.txt'.format(prefix,prefix_2021)),is_train=False,is_eval=True)
    #     #file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'protocols/wild_meta.txt'.format(prefix,prefix_2021)),is_train=False,is_eval=True)

    #     print('no. of eval trials',len(file_eval))
    #     #eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_eval/'.format(args.track)))
    #     eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+''.format(args.track)))
    #     produce_evaluation_file(eval_set, model, device, args.eval_output)
    #     sys.exit(0)

    
    # define train dataloader
    d_label_trn, file_train = genSpoof_list_multidata(dir_meta =  os.path.join(args.protocols_path, train_protocol_filename), is_train=True, is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    # train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)

    train_set=Multi_Dataset_train(args, list_IDs=file_train, labels = d_label_trn, base_dir=args.database_path, algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn

    # define validation dataloader
    d_label_dev,file_dev = genSpoof_list_multidata(dir_meta=os.path.join(args.protocols_path, dev_protocol_filename), is_train=False, is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Multi_Dataset_train(args, list_IDs=file_dev, labels=d_label_dev, base_dir=args.database_path, algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    best_val_acc = 0.5
    n_swa_update = 0
    
    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader, model, args.lr, optimizer, device)
        
        val_loss, val_balanced_acc = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        writer.add_scalar('val_balanced_acc', val_balanced_acc, epoch)
        print('\n{} - {} - {} - {} '.format(epoch, running_loss, val_loss, val_balanced_acc))
        
        if best_val_acc <= val_balanced_acc:
            print("best model find at epoch", epoch)
            best_val_acc = val_balanced_acc
            torch.save(model.state_dict(), os.path.join(model_save_path, "epoch_{}_{:03.3f}.pth".format(epoch, val_balanced_acc)))

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1

        writer.add_scalar("best val balanced accuracy", best_val_acc, epoch)


    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(train_loader, model, device=device)

    torch.save(model.state_dict(), os.path.join(model_save_path, "swa.pth"))
    
    