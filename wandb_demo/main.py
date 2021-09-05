import torch
import torchvision
import wandb
from sweep_config import sweep_config
from mnist import MNISTDataset

sweep_id = wandb.sweep(sweep_config, project="Pytorch-sweeps")


def build_dataset(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset = MNISTDataset(transform, train=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return train_loader


def train():
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'epochs': 5,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'fc_layer_size': 128,
        'dropout': 0.5,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    # Define the model architecture - This is a simplified version of the VGG19 architecture
    network = torch.nn.Sequential(torch.nn.Flatten(start_dim=1),
                                  torch.nn.Linear(784, config.fc_layer_size),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(config.dropout),
                                  torch.nn.Linear(config.fc_layer_size, 10),
                                  torch.nn.LogSoftmax(dim=1))
    train_loader = build_dataset(config.batch_size)
    # Set of Conv2D, Conv2D, MaxPooling2D layers with 32 and 64 filters

    # Define the optimizer
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=config.learning_rate,
                                    momentum=0.9)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=config.learning_rate)

    network.train()
    network = network.to(device)
    for i in range(config.epochs):
        closs = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.long().to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = torch.torch.nn.functional.nll_loss(output, target)
            loss.backward()
            closs = closs + loss.item()
            optimizer.step()
            wandb.log({"batch loss": loss.item()})
        wandb.log({"loss": closs / config.batch_size})

wandb.agent(sweep_id, train)