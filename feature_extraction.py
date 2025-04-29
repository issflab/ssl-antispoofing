import os
import pickle
import numpy as np
import torch
import torchaudio
from s3prl.nn import S3PRLUpstream, Featurizer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split


class deep_learning:

    def __init__(self, model_name, device='cpu'):

        self.SAMPLE_RATE = 16000
        self.MAX_SECONDS = 60
        self.model = S3PRLUpstream(model_name).to(device)
        self.featurizer = Featurizer(self.model).to(device)
        self.device = device

    def extract_features(self, file_path, aggregate_emb=False, layer_number=0):
        """
        Extract features from an audio file using the loaded S3PRL model.
        
        Args:
            file_path (str): Path to the audio file.
            aggregate_emb (bool): 
                - If True, returns a single aggregated embedding across layers.
                - If False, returns the embedding(s) from the model's hidden layers.
            layer_number (int, optional): When aggregate_emb is False, if a specific 
                layer is desired, specify its index (0-indexed). If not provided, all 
                layer embeddings are returned.
        
        Returns:
            numpy.ndarray: 
                - Aggregated embedding if aggregate_emb=True.
                - Otherwise, either the specified layer’s embedding or a stack of all 
                  layer embeddings.
        """
        # Load the audio file and obtain metadata.
        waveform, sample_rate = torchaudio.load(file_path)

        # adjust for the different sampling rates
        if sample_rate != self.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.SAMPLE_RATE)

        waveform = waveform/ waveform.abs().max()
        metadata = torchaudio.info(file_path)

        
        # Ensure waveform has shape [batch, samples]
        if waveform.ndimension() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute maximum length and pad/truncate accordingly to the max_length of 60s.
        max_length = int(self.SAMPLE_RATE * self.MAX_SECONDS)
        padded_waveform = torch.zeros(waveform.size(0), max_length)

        for i, wav in enumerate(waveform):
            end = min(max_length, wav.size(0))
            padded_waveform[i, :end] = wav[:end]
        wavs_len = torch.LongTensor([min(max_length, waveform.size(1)) for _ in range(waveform.size(0))])
        
        # Forward pass through the model.
        with torch.no_grad():
            all_hs, all_hs_len = self.model(padded_waveform.to(self.device), wavs_len.to(self.device))
        
        if aggregate_emb:
            # Compute aggregated embedding: average the mean-pooled embeddings of all layers.
            embedding = self.aggregate_embeddings(all_hs)
            return embedding.cpu().numpy()
        
        else:
            # Get embeddings for each layer: mean pooling over the time dimension.
            layer_embeddings = [layer.mean(dim=1).cpu().numpy() for layer in all_hs]
            if layer_number is not None:
                if layer_number < 0 or layer_number >= len(layer_embeddings):
                    raise ValueError(f"Invalid layer_number {layer_number}. Must be between 0 and {len(layer_embeddings)-1}.")
                return layer_embeddings[layer_number]
            else:
                return np.stack(layer_embeddings, axis=0)

    def aggregate_embeddings(self, all_hs):
        """
        Aggregates embeddings from all hidden layers by computing the mean over time for 
        each layer and then averaging across layers.
        
        Args:
            all_hs (list of torch.Tensor): List of hidden state tensors.
        
        Returns:
            torch.Tensor: Aggregated embedding of shape [1, n_features].
        """
        embeddings_list = [layer.mean(dim=1) for layer in all_hs]
        final_embedding = sum(embeddings_list) / len(embeddings_list)
        return final_embedding

    def extract_feat_from_waveform(self, waveform_tensor, aggregate_emb=False, layer_number=0):
        waveform_tensor = waveform_tensor / waveform_tensor.abs().max()

        if waveform_tensor.ndimension() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)

        max_length = int(self.SAMPLE_RATE * self.MAX_SECONDS)
        padded_waveform = torch.zeros(waveform_tensor.size(0), max_length)

        for i, wav in enumerate(waveform_tensor):
            end = min(max_length, wav.size(0))
            padded_waveform[i, :end] = wav[:end]
        wavs_len = torch.LongTensor([min(max_length, waveform_tensor.size(1)) for _ in range(waveform_tensor.size(0))])

        with torch.no_grad():
            all_hs, all_hs_len = self.model(padded_waveform.to(self.device), wavs_len.to(self.device))

        if aggregate_emb:
            embedding = self.aggregate_embeddings(all_hs)
            return embedding.cpu().numpy()
        else:
            layer_embeddings = [layer.mean(dim=1).cpu().numpy() for layer in all_hs]
            if layer_number is not None:
                if layer_number < 0 or layer_number >= len(layer_embeddings):
                    raise ValueError(f"Invalid layer_number {layer_number}. Must be between 0 and {len(layer_embeddings)-1}.")
                return layer_embeddings[layer_number]
            else:
                return np.stack(layer_embeddings, axis=0)
