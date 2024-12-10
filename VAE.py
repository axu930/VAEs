import torch
import torch.nn as nn
import torch.nn.functional as F

# most basic encoder/decoder for MNIST, 2 hidden layers

class VAE(nn.Module):

    def __init__(self,latent_dim, beta, dropout = 0.0):
        super().__init__()
        self.enc = encoder(latent_dim, dropout)
        self.dec = decoder(latent_dim, dropout)
        self.latent_dim = latent_dim
        self.beta = beta

    def forward(self, x, debug = False):
        means, log_stds = self.enc(x)
        stds = torch.exp(log_stds)

        noise = torch.randn_like(stds)
        z = means + noise * stds 
        if debug:
            print(f"means: {means}, log stds: {log_stds}")

        out = self.dec(z)

        mse_loss = F.mse_loss(x,out, reduction='mean')

        # KL divergence of 2 normal distributions is the last formula in
        # https://stanford.edu/~jduchi/projects/general_notes.pdf
        # sigma2 in the formula is std normal

        kl_loss = torch.sum(-log_stds + stds + means ** 2, axis = 1) 
        kl_loss = (kl_loss - self.latent_dim) / 2
        kl_loss = torch.mean(kl_loss)

        loss =  mse_loss + self.beta * kl_loss / 784 #784 pixels

        return out, loss
    


class encoder(nn.Module):

    def __init__(self, latent_dim, dropout):
        super().__init__()
        self.latent_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(784, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64,latent_dim * 2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B = x.size()[0]

        x = x.view(B,784)
        x = self.mlp(x)

        means, log_stds = x.split([self.latent_dim, self.latent_dim], dim = -1)

        return means, log_stds


class decoder(nn.Module):

    def __init__(self, latent_dim, dropout):
        super().__init__()
        self.latent_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64,256),
            nn.GELU(),
            nn.Linear(256,784),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        B = x.size()[0]
        x = self.mlp(x)
        x = x.view(B,1,28,28)

        return F.sigmoid(x)

"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout = 0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout) #Use dropout regularization
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x
    """