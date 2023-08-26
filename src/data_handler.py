import torchvision
import numpy as np
import torch
import torchvision.transforms as tt


import os
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetDataset(Dataset):
    def __init__(self, root, train=False, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.data_folder = os.path.join(self.root, 'train')
        else:
            self.data_folder = os.path.join(self.root, 'val', 'images')
            self.labels_file = os.path.join(
                self.root, 'val', 'val_annotations.txt')
            self.labels = self._load_labels()

        self.image_paths = sorted(os.listdir(self.data_folder))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.data_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.mode == 'val':
            label = self.labels[image_name]
        else:
            label = image_name.split('_')[0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def _load_labels(self):
        labels = {}
        with open(self.labels_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                labels[line[0]] = line[1]
        return labels


class DataWrapper:
    def __init__(self, name, batch_size):
        if name == "cifar":
            train_data = torchvision.datasets.CIFAR100(
                './data', train=True, download=True)
        elif name == "tiny":
            train_data = TinyImageNetDataset(
                './data/tiny-imagenet-200', train=True)
        x = np.concatenate([np.asarray(train_data[i][0])
                           for i in range(len(train_data))])
        # calculate the mean and std along the (0, 1) axes
        mean = np.mean(x, axis=(0, 1))/255
        std = np.std(x, axis=(0, 1))/255
        # the the mean and std
        mean = mean.tolist()
        std = std.tolist()
        transform_train = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                                      tt.RandomHorizontalFlip(),
                                      tt.ToTensor(),
                                      tt.Normalize(mean, std, inplace=True)])
        transform_test = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
        if name == "cifar":
            trainset = torchvision.datasets.CIFAR100("./data",
                                                     train=True,
                                                     download=True,
                                                     transform=transform_train)
            testset = torchvision.datasets.CIFAR100("./data",
                                                    train=False,
                                                    download=True,
                                                    transform=transform_test)
        elif name == "tiny":
            trainset = TinyImageNetDataset("./data/tiny-imagenet-200",
                                           train=True,
                                           transform=transform_train)
            testset = TinyImageNetDataset("./data/tiny-imagenet-200",
                                          train=False,
                                          transform=transform_test)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size, pin_memory=True, num_workers=2)
        device = get_default_device()
        self.trainloader = DeviceDataLoader(self.trainloader, device)
        self.testloader = DeviceDataLoader(self.testloader, device)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
