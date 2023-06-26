import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision import transforms


class DiffSet(Dataset):
    def __init__(self, train, dataset_name="MNIST"):

        ds_maping = {
            "MNIST": MNIST,
            "FashionMNIST": FashionMNIST,
            "CIFAR10": CIFAR10,
            "CelebA": CelebA,
        }

        img_sizes = {
            "MNIST": 32,
            "FashionMNIST": 32,
            "CIFAR10": 32,
            "CelebA": 128
        }

        if dataset_name != "CelebA":
            t = transforms.Compose([transforms.ToTensor()])
            ds = ds_maping[dataset_name](
                "./data", download=True, train=train, transform=t
            )
        else:
            t = transforms.Compose([transforms.ToTensor(), transforms.Resize((img_sizes[dataset_name], img_sizes[dataset_name]), antialias=True)])
            ds = ds_maping[dataset_name](
                "./data", download=True, transform=t, split=("train" if train else "test")
            )

        self.ds = ds
        self.dataset_name = dataset_name
        self.size = img_sizes[dataset_name]
 
        # Set channel and image size
        if self.dataset_name == "MNIST" or self.dataset_name == "FashionMNIST":
            self.depth = 1 # MNIST only has 1 channel (grayscale)
        elif self.dataset_name == "CIFAR10":
            self.depth = 3 # 3 channels (RGB)
        elif self.dataset_name == "CelebA":
            self.depth = 3 # 3 channels (RGB)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        ds_item = self.ds[item][0]

        if self.dataset_name == "MNIST" or self.dataset_name == "FashionMNIST":
            pad = transforms.Pad(2)
            data = pad(ds_item) # Pad to make it 32x32
            # Add a channel dimension, because the original data only has 1 channel
        elif self.dataset_name == "CIFAR10":
            data = torch.Tensor(ds_item)
        elif self.dataset_name == "CelebA":
            data = torch.Tensor(ds_item)
        
        data = ((data / 255.0) * 2.0) - 1.0 # normalize to [-1, 1]
        # Move the channel dimension as second dimension
        # data = data.moveaxis(3, 1)  # (N, H, W, C) -> (N, C, H, W)

        return data