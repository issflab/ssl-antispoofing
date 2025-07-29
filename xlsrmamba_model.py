from feature_extraction import deep_learning
from mamba_blocks import MixerModel
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = deep_learning(model_name=args.ssl_feature, device=device)
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
