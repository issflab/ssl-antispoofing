from __future__ import annotations

import torch
import torch.nn as nn
from s3prl.nn import Featurizer, S3PRLUpstream


class SSLFrontend(nn.Module):
    """Shared S3PRL-based SSL frontend used by multiple backend models."""

    def __init__(
        self,
        n_layers: int,
        device,
        model_name: str,
        freeze_ssl: bool = True,
        feature_mode: str = "featurizer",
        layer_index: int = -1,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.model = S3PRLUpstream(self.model_name).to(self.device)
        self.featurizer = Featurizer(self.model).to(self.device)
        self.n_layers = n_layers
        self.out_dim = self.featurizer.output_size
        self.freeze_ssl = freeze_ssl
        self.feature_mode = feature_mode
        self.layer_index = layer_index

    def _get_lengths(self, waveform):
        return torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])

    def _forward_upstream(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = self._get_lengths(waveform)
        if self.freeze_ssl:
            with torch.no_grad():
                return self.model(waveform.to(self.device), wavs_len.to(self.device))
        return self.model(waveform.to(self.device), wavs_len.to(self.device))

    def _normalize_hidden_state(self, hidden_state):
        return hidden_state[0].permute(1, 0, 2) if isinstance(hidden_state, tuple) else hidden_state

    def _normalized_hidden_states(self, waveform):
        all_hs, _ = self._forward_upstream(waveform)
        return [self._normalize_hidden_state(hidden_state) for hidden_state in all_hs]

    def _resolve_layer_index(self, total_layers: int) -> int:
        idx = self.layer_index
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise ValueError(
                f"Invalid frontend layer index {self.layer_index} for {total_layers} SSL layers"
            )
        return idx

    def extract_feat_featurizer(self, waveform):
        all_hs, all_hs_len = self._forward_upstream(waveform)
        hs, hs_len = self.featurizer(all_hs, all_hs_len)
        return hs, hs_len

    def extract_feat_stack(self, waveform):
        all_hs, _ = self._forward_upstream(waveform)
        return torch.stack(
            [t[0].permute(1, 0, 2) if isinstance(t, tuple) else t for t in all_hs[: self.n_layers]],
            dim=1,
        )

    def extract_hidden_states(self, waveform):
        hidden_states = self._normalized_hidden_states(waveform)

        if self.feature_mode == "all_hidden_states":
            return hidden_states
        if self.feature_mode == "specific_layer":
            return [hidden_states[self._resolve_layer_index(len(hidden_states))]]
        if self.feature_mode == "featurizer":
            sequence_features, _ = self.extract_feat_featurizer(waveform)
            return [sequence_features]

        raise ValueError(f"Unsupported frontend feature mode for hidden states: {self.feature_mode}")

    def extract_sequence_features(self, waveform):
        if self.feature_mode == "featurizer":
            sequence_features, _ = self.extract_feat_featurizer(waveform)
            return sequence_features

        hidden_states = self._normalized_hidden_states(waveform)

        if self.feature_mode == "specific_layer":
            return hidden_states[self._resolve_layer_index(len(hidden_states))]
        if self.feature_mode == "all_hidden_states":
            raise ValueError(
                "frontend_feature_mode='all_hidden_states' is not valid for backends expecting a single sequence feature map"
            )

        raise ValueError(f"Unsupported frontend feature mode: {self.feature_mode}")

    def _sample_indices(self, total_layers: int):
        k = min(self.n_layers, total_layers)
        if k == total_layers:
            return list(range(total_layers))
        step = (total_layers - 1) / (k - 1)
        return [int(step * i) for i in range(k)]

    def extract_feat_sample(self, waveform):
        all_hs, _ = self._forward_upstream(waveform)
        idxs = self._sample_indices(len(all_hs))
        feats = []
        for i in idxs:
            t = all_hs[i]
            x = t[0].permute(1, 0, 2) if isinstance(t, tuple) else t
            feats.append(x)
        return torch.stack(feats, dim=1)

    def extract_feat_1n(self, waveform):
        all_hs, _ = self._forward_upstream(waveform)
        return torch.stack(
            [t[0].permute(1, 0, 2) if isinstance(t, tuple) else t for t in all_hs[1 : self.n_layers + 1]],
            dim=1,
        )

    def freeze_feature_extraction(self):
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
