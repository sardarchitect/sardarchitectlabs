import torch
import idx2numpy
import os

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=False):
        self.transform = transform
        self.train = train
        dataroot = '/SardarchitectLabs/data/MNIST/raw/'
        if train:
            img_dir = os.path.join(dataroot,'train-images-idx3-ubyte')
            lbl_dir = os.path.join(dataroot,'train-labels-idx1-ubyte')
        else:
            img_dir = os.path.join(dataroot,'t10k-images-idx3-ubyte')
            lbl_dir = os.path.join(dataroot,'t10k-labels-idx1-ubyte')
        
        self.images = idx2numpy.convert_from_file(img_dir)
        self.labels = idx2numpy.convert_from_file(lbl_dir)
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        
        return img, lbl