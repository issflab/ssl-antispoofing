from mamba_blocks import MixerModel
import torch.nn as nn
import torch
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

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(24, device=self.device, args=self.args)
        self.linear_proj = nn.Linear(self.ssl_model.out_dim, args.emb_size)

        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)

        self.conformer = MixerModel(
            d_model=args.emb_size,
            n_layer=args.num_encoders // 2,
            ssm_cfg={},
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True
        )

    def forward(self, x):
        x_feat = torch.tensor(
    self.ssl_model.extract_feat_from_waveform(x, aggregate_emb=False),
    dtype=torch.float32,
    device=self.device
)
        x_feat = self.linear_proj(x_feat)
        x_feat = x_feat.unsqueeze(1)
        x_feat = self.first_bn(x_feat)
        x_feat = self.selu(x_feat)
        x_feat = x_feat.squeeze(1)
        return self.conformer(x_feat)
