import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from feature_extraction import deep_learning



class SSLModel(nn.Module):
    def __init__(self, device, args):
        super(SSLModel, self).__init__()

        self.device = device
        self.layer_num = 0

        self.extractor = deep_learning(model_name=args.ssl_feature, device=device)

        out_dim_dict = {'wavlm_large': 1024, 'mae_ast_frame':768, 'npc_960hr':512}
        self.out_dim = out_dim_dict[args.ssl_feature]

        # self.device = device
        # self.extractor = deep_learning("hubert", device=self.device)  # or "wav2vec2_base", etc.
        # self.out_dim = 1024  # Typical dim for HuBERT/Wav2Vec2.0 large
        # return

    def extract_feat(self, input_data):
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        # if input_data.ndim == 3:
        #     input_data = input_data[:, :, 0]  
        #     # remove channel dim
        # else:
        #     input_data = input_data

        # Normalize
        # input_data = input_data / input_data.abs().max()

        # Extract all layers
        layer_outputs = self.extractor.extract_feat_from_waveform(input_data, aggregate_emb=False, layer_number=None)  # List of (B, T, D)
        
        # Aggregate output (you can also extract a specific layer instead if needed)
        # emb = self.extractor.aggregate_embeddings([torch.tensor(l).to(self.device) for l in layer_outputs])
        
        # Convert all to torch tensors
        layerresult = [torch.tensor(l).to(self.device) for l in layer_outputs]

        # print(layerresult.shape)
        # print(len(layerresult))
        # print(layerresult[0].shape)

        # return emb, layerresult
        return layerresult


def getAttenF(layerResult):
    """
    Extracts pooled and full-resolution features from a list of hidden states.

    Args:
        layerResult (List[Tensor]): A list of hidden states from each transformer layer.
            Each Tensor should have shape [B, T, D], where:
                - B: Batch size
                - T: Sequence length
                - D: Feature dimension (e.g., 768 or 1024)

    Returns:
        layery (Tensor): Layer-wise pooled features of shape [B, num_layers, D].
            This is obtained by applying adaptive average pooling across time for each layer.

        fullfeature (Tensor): Full hidden state maps stacked across layers with shape [B, num_layers, T, D].
            This retains the complete temporal resolution for each layer.
    """
    poollayerResult = []
    fullf = []

    for layer in layerResult:
        # layer: [B, T, D]
        layery = layer.transpose(1, 2)               # [B, D, T]
        layery = F.adaptive_avg_pool1d(layery, 1)    # [B, D, 1]
        layery = layery.transpose(1, 2)              # [B, 1, D]
        poollayerResult.append(layery)

        x = layer.unsqueeze(1)                       # [B, 1, T, D]
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)       # [B, num_layers, D]
    fullfeature = torch.cat(fullf, dim=1)            # [B, num_layers, T, D]

    return layery, fullfeature


# def getAttenF(layer_embeddings):
#     """
#     Args:
#         layer_embeddings (List[Tensor]): List of layer outputs from S3PRL, each of shape (B, T, D)
    
#     Returns:
#         layery: (B, num_layers, D)
#         fullfeature: (B, num_layers, D, T)
#     """
#     poollayerResult = []
#     fullf = []

#     for layer in layer_embeddings:
#         # Ensure tensor and correct shape: (B, T, D)
#         if isinstance(layer, np.ndarray):
#             layer = torch.tensor(layer)
#         #print(f"Layer shape = {layer.shape}, type = {type(layer)}")

#         if layer.ndim != 3:
#             raise ValueError("Each layer must be a 3D tensor (B, T, D)")

#         # Apply adaptive pooling to squeeze time axis to 1
#         layery = F.adaptive_avg_pool1d(layer.permute(0, 2, 1), 1)  # (B, D, 1)
#         layery = layery.permute(0, 2, 1)  # (B, 1, D)
#         poollayerResult.append(layery)

#         # For attention weighting
#         x = layer.permute(0, 2, 1).unsqueeze(1)  # (B, 1, D, T)
#         fullf.append(x)

#     layery = torch.cat(poollayerResult, dim=1)  # (B, num_layers, D)
#     fullfeature = torch.cat(fullf, dim=1)       # (B, num_layers, D, T)
#     return layery, fullfeature


# def getAttenF(layerResult):
#     poollayerResult = []
#     fullf = []
#     for layer in layerResult:

#         print(layer.shape)
#         layery = layer[0].transpose(0, 1).transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
#         layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
#         layery = layery.transpose(1, 2) # (b,1,1024)
#         poollayerResult.append(layery)

#         x = layer[0].transpose(0, 1)
#         x = x.view(x.size(0), -1,x.size(1), x.size(2))
#         fullf.append(x)

#     layery = torch.cat(poollayerResult, dim=1)
#     fullfeature = torch.cat(fullf, dim=1)
#     return layery, fullfeature


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        # self.ssl_extractor = deep_learning("hubert", device=self.device)
        self.ssl_model = SSLModel(self.device, args)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(self.ssl_model.out_dim, 1)
        self.sig = nn.Sigmoid()
        # self.fc1 = nn.Linear(17152, 1024)
        self.fc1 = None  # will initialize during first forward
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, x):
        # Debug shape
        #print(f"[DEBUG] Input shape to Model.forward(): {x.shape}")
        # Step 1: Ensure shape (B, T)
        # if x.ndim == 3:
        #     if x.shape[2] == 1:  # (B, T, 1)
        #         x = x.squeeze(-1)
        #     elif x.shape[0] == 1 and x.shape[1] > 1:  # (1, B, T) → (B, T)
        #         x = x.squeeze(0)
        #     elif x.shape[0] > 1 and x.shape[1] > 1:
        #         raise ValueError(f"Ambiguous 3D input: {x.shape}")
        # elif x.ndim == 2:
        #     pass  # Already (B, T)
        # elif x.ndim == 1:
        #     x = x.unsqueeze(0)
        # else:
        #     raise ValueError(f"Unexpected input shape: {x.shape}")

        #print(f"[DEBUG] Final input shape for S3PRL: {x.shape}")

        # Step 2: Feature extraction
        # layer_embeddings = self.ssl_extractor.extract_feat_from_waveform(x, aggregate_emb=False)
        # x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)

        # Step 3: Attention over layers
        y0, fullfeature = getAttenF(layerResult)

        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        # print("Flattened size:", x.shape)

        # 🔧 Lazy initialization here
        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 1024).to(self.device)

        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)

        return output

