import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision import transforms


class DiffSet(Dataset):
    def __init__(self, train, dataset_name):

        ds_mapping = {
            "MNIST": (MNIST, 32, 1),
            "FashionMNIST": (FashionMNIST, 32, 1),
            "CIFAR10": (CIFAR10, 32, 3),
        }

        t = transforms.Compose([transforms.ToTensor()])
        ds, img_size, channels = ds_mapping[dataset_name]
        ds = ds("./data", download=True, train=train, transform=t)

        self.ds = ds
        self.dataset_name = dataset_name
        self.size = img_size
        self.depth = channels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        ds_item = self.ds[item][0]

        if self.dataset_name == "MNIST" or self.dataset_name == "FashionMNIST":
            pad = transforms.Pad(2)
            data = pad(ds_item) # Pad to make it 32x32
        else:
            data = ds_item
        
        data = (data * 2.0) - 1.0 # normalize to [-1, 1].
        return data