import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CNN = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25)
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(9216, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 10)
        )
        
    def forward(self, x):
        out = self.CNN(x)
        out = out.reshape(out.shape[0],-1)
        out = self.linear(out)
        out = torch.nn.functional.log_softmax(out, dim=1)
        return out