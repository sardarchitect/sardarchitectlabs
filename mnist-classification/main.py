# IMPORT PACKAGES
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
# IMPORT CUSTOM MODULES
from mnist import MNISTDataset
from models import CNN

def train(dataloader, model, criterion, optimizer, device):
    start_time = time.time()
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (X, y) in enumerate(dataloader):        
        X = X.to(device)
        y = y.long().to(device)
        
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = out.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
                
    train_loss = round((running_loss/len(dataloader.dataset)), 4)
    acc = round((100. * correct/total), 2)
        
    print(f"Train Loss: {train_loss} | Accuracy: {acc}")
    print(f'Duration: {time.time() - start_time:.0f} seconds')

    return train_loss, acc

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Classifier')
    parser.add_argument('--batch-size', type=int, default=32, metavar='', help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=int, default=1e-3, metavar='', help='Learning rate for training (default: 1e-3)')
    parser.add_argument('--num-epoch', type=int, default=5, metavar='', help='Number of Epoch (default: 25)')
    parser.add_argument('--device', type=str, default='cpu', metavar='', help='Define device (default: cpu)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_ds = MNISTDataset(transform, train=True)
    test_ds = MNISTDataset(transform, train=False)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = args.batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size = args.batch_size, shuffle=False)

    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_acc = []
    train_losses = []
    
    start_time = time.time()
    for epoch in range(args.num_epoch):
        train_loss, acc = train(train_dl, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_acc.append(acc)
    print("Done!")
    print(f'\nTOTAL DURATION: {time.time() - start_time:.0f} seconds')

if __name__ == '__main__':
    main()