import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset  # 数据加载
import numpy as np
from typing import List
from torch.optim import AdamW


class Encoder(nn.Module):
    """A class that encapsulates the encoder."""
    def __init__(
        self,
        num_genes: int,
        latent_dim: int = 64,
        hidden_dim: List[int] = [128, 128],
        dropout: float = 0,
        input_dropout: float = 0,
        residual: bool = False
            ,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(num_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x) -> torch.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)
        state_dict = ckpt['state_dict']
        first_layer_key = ['network.0.1.weight',
            'network.0.1.bias',
            'network.0.2.weight',
            'network.0.2.bias',
            'network.0.2.running_mean',
            'network.0.2.running_var',
            'network.0.2.num_batches_tracked',
            'network.0.3.weight]',]
        for key in first_layer_key:
            if key in state_dict:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

class Decoder(nn.Module):
    """A class that encapsulates the decoder."""
    def __init__(
        self,
        num_genes: int,
        latent_dim: int = 64,
        hidden_dim: List[int] = [128, 128],
        dropout: float = 0,
        residual: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # other hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # reconstruction layer
        self.network.append(nn.Linear(hidden_dim[-1], num_genes))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def save_state(self, filename: str):
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)
        state_dict = ckpt['state_dict']
        last_layer_key = ['network.3.weight', 'network.3.bias']
        for key in last_layer_key:
            if key in state_dict:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

class VAE(nn.Module):
    """VAE base on compositional perturbation autoencoder (CPA)"""
    def __init__(
        self,
        num_genes: int,
        latent_dim: int = 64,
        device: str = "cuda",
        seed: int = 0,
        loss_ae: str = "mse",
        decoder_activation: str = "linear",
        hidden_dim: List[int] = [128, 128, 128],
    ):
        super(VAE, self).__init__()
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.best_score = -1e5
        self.patience_trials = 0

        self.set_hparams_(hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = 0.1
        self.input_dropout = 0.1
        self.residual = False
        self.encoder = Encoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
        )
        self.decoder = Decoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
            residual=self.residual,
        )
        self.loss_autoencoder = nn.MSELoss(reduction='mean')
        self.iteration = 0
        self.to(self.device)
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder, True)
            + get_params(self.decoder, True)
        )
        self.optimizer_autoencoder = AdamW(_parameters, lr=self.hparams["autoencoder_lr"], weight_decay=self.hparams["autoencoder_wd"])

    def forward(self, genes, return_latent=False, return_decoded=False):
        if return_decoded:
            gene_reconstructions = self.decoder(genes)
            gene_reconstructions = nn.ReLU()(gene_reconstructions)
            return gene_reconstructions
        latent_basal = self.encoder(genes)
        if return_latent:
            return latent_basal
        gene_reconstructions = self.decoder(latent_basal)
        return gene_reconstructions

    def set_hparams_(self, hidden_dim):
        self.hparams = {
            "dim": self.latent_dim,
            "autoencoder_width": 5000,
            "autoencoder_depth": 3,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 0.0001,
            "autoencoder_lr": 5e-4,
        }
        return self.hparams

    def train_vae(self, genes):
        genes = genes.to(self.device)
        gene_reconstructions = self.forward(genes)
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)
        total_loss = reconstruction_loss
        self.optimizer_autoencoder.zero_grad()
        total_loss.backward()
        self.optimizer_autoencoder.step()
        self.iteration += 1
        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "total_loss": total_loss.item(),
        }

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, time_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_features, out_features),
        )

class AE(nn.Module):
    ''' Autoencoder for dimensional reduction'''
    def __init__(self, dim):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(512, 128)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(128, 512)
        self.prelu3 = nn.PReLU()
        self.fc4 = nn.Linear(512, dim)
        self.prelu4 = nn.PReLU()

    def encode(self, x):
        h1 = self.prelu1(self.fc1(x))
        return self.prelu2(self.fc2(h1))

    def decode(self, z):
        h3 = self.prelu3(self.fc3(z))
        return self.prelu4(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z

def train_autoencoder(autoencoder, data, epochs=100, learning_rate=0.0001):
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    data = torch.tensor(data, dtype=torch.float32)

    for epoch in range(epochs):
        autoencoder.train()
        optimizer.zero_grad()
        reconstructed, _ = autoencoder(data)
        loss = criterion(reconstructed, data)
        loss.backward()
        optimizer.step()

        #if (epoch + 1) % 10 == 0:
            #print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return autoencoder