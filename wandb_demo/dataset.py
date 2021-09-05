import torch
# import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.nn as nn
import wandb
from torchvision import datasets, transforms
from sweep_config import sweep_config
from mnist import MNISTDataset

sweep_id = wandb.sweep(sweep_config, project="Pytorch-sweeps")


def build_dataset(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
    dataset = MNISTDataset(transform, train=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return train_loader

ds = build_dataset(5)
print(ds)