import torch
from torch.utils.data import TensorDataset

from pytorch_lightning import LightningDataModule


class SyntheticGmm(LightningDataModule):
    name = 'magic'

    def __init__(self, input_size, train_length, val_length, mean_distance_scale, batch_size,
                 seed=42, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.train_length = train_length
        self.val_length = val_length
        self.mean_distance_scale = mean_distance_scale
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage=None):
        d = self.input_size
        N = self.train_length + self.val_length
        r = self.mean_distance_scale
        self.mu = r * torch.stack((torch.ones(d), -torch.ones(d)))

        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        Z = torch.randint(2, (N, ))
        X = torch.randn(N, d) + self.mu[Z]

        dot = self.mu @ X.t()
        pstar = torch.softmax(dot, dim=0)
        # p_unnormalized = torch.exp(-torch.norm(X - mu[:, None], dim=-1)**2 / 2)
        # p_normalized = p_unnormalized / p_unnormalized.sum(dim=0)
        # assert torch.allclose(p_normalized, pstar)

        Y = (torch.rand(N) < pstar[1]).long()
        torch.set_rng_state(rng_state)

        self.train = TensorDataset(X[:self.train_length], Y[:self.train_length])
        self.val = TensorDataset(X[self.train_length:], Y[self.train_length:])

    def train_dataloader(self):
        # Much faster to set num_workers=0
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                                           num_workers=0)

    def val_dataloader(self):
        # Much faster to set num_workers=0
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                                           num_workers=0)
