import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from s3prl.nn import S3PRLUpstream, Featurizer



class SSLModel(nn.Module):
    def __init__(self, n_layerss, device, args):
        super(SSLModel, self).__init__()
        self.device = device
        self.model_name = args.ssl_model
        self.model = S3PRLUpstream(self.model_name).to(self.device)
        self.featurizer = Featurizer(self.model).to(self.device)
        self.n_layers=n_layerss
        self.out_dim = self.featurizer.output_size

    def extract_feat_featurizer(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])
        with torch.no_grad():
            all_hs, all_hs_len = self.model(waveform.to(self.device), wavs_len.to(self.device))
        hs, hs_len = self.featurizer(all_hs, all_hs_len)
        return hs, hs_len
    
    def extract_feat(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])
        with torch.no_grad():
            all_hs, all_hs_len = self.model(waveform.to(self.device), wavs_len.to(self.device))
        return torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in all_hs[:self.n_layers]], dim=1)
    
    def _sample_indices(self, total_layers: int):
        k = min(self.n_layers, total_layers)
        if k == total_layers:
            return list(range(total_layers))
        step = (total_layers - 1) / (k - 1)
        return [int(step * i) for i in range(k)]

    def extract_feat_sample(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1)] * waveform.size(0))
        with torch.no_grad():
            all_hs, _ = self.model(waveform.to(self.device), wavs_len.to(self.device))
        # sample your indices
        idxs = self._sample_indices(len(all_hs))
        # print(idxs)
        # pick & permute
        feats = []
        for i in idxs:
            t = all_hs[i]
            x = t[0].permute(1,0,2) if isinstance(t, tuple) else t
            feats.append(x)
        # result: (batch, chosen_layers, time, dim)
        # print(torch.stack(feats, dim=1).shape)
        return torch.stack(feats, dim=1)
    
    def extract_feat_1n(self, waveform):
        # print(waveform.shape,wavs_len.shape)
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])
        # print(waveform.shape,wavs_len.shape)
        with torch.no_grad():
            all_hs, all_hs_len = self.model(waveform.to(self.device), wavs_len.to(self.device))
        return torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in all_hs[1:self.n_layers + 1]], dim=1)

    def freeze_feature_extraction(self):
        """Freezes the feature extraction layers of the base SSL model."""
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False


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
        self.ssl_model = SSLModel(24, device=self.device, args=self.args)

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

