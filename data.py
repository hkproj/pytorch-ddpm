import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision import transforms


class DiffSet(Dataset):
    def __init__(self, train, dataset_name="MNIST"):

        ds_mapping = {
            "MNIST": (MNIST, 32),
            "FashionMNIST": (FashionMNIST, 32),
            "CIFAR10": (CIFAR10, 32),
            "CelebA": (CelebA, 128),
        }

        ds, img_size = ds_mapping[dataset_name]

        if dataset_name != "CelebA":
            t = transforms.Compose([transforms.ToTensor()])
            ds = ds(
                "./data", download=True, train=train, transform=t
            )
        else:
            t = transforms.Compose([transforms.ToTensor(), transforms.Resize((img_size, img_size), antialias=True)])
            ds = ds(
                "./data", download=True, transform=t, split=("train" if train else "test")
            )

        self.ds = ds
        self.dataset_name = dataset_name
        self.size = img_size
 
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
        else:
            data = ds_item
        
        # No need to scale by 255 because it is already in [0, 1] range thanks to the ToTensor transform
        data = (data * 2.0) - 1.0 # normalize to [-1, 1].
        return data