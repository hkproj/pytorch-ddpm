import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms


class DiffSet(Dataset):
    def __init__(self, train, dataset="MNIST"):
        transform = transforms.Compose([transforms.ToTensor()])

        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
        }

        train_dataset = datasets[dataset](
            "./data", download=True, train=train, transform=transform
        )

        self.dataset_len = len(train_dataset.data)

        if dataset == "MNIST" or dataset == "Fashion":
            pad = transforms.Pad(2)
            data = pad(train_dataset.data) # Pad to make it 32x32
            # Add a channel dimension, because the original data only has 1 channel
            data = data.unsqueeze(3) # (N, H, W) -> (N, H, W, C)
            self.channels = 1 # MNIST only has 1 channel (grayscale)
            self.size = 32
        elif dataset == "CIFAR":
            data = torch.Tensor(train_dataset.data) # (N, H, W, C)
            self.channels = 3 # 3 channels (RGB)
            self.size = 32
        self.input_seq = ((data / 255.0) * 2.0) - 1.0 # normalize to [-1, 1]
        # Move the channel dimension as second dimension
        self.input_seq = self.input_seq.moveaxis(3, 1)  # (N, H, W, C) -> (N, C, H, W)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item]

if __name__ == '__main__':
    ds = DiffSet(True, "Fashion")
    print(ds[0].shape)