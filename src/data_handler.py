import torchvision
import numpy as np
import torch
import torchvision.transforms as tt


class DataWrapper:
    def __init__(self, name, batch_size):
        if name == "cifar":
            train_data = torchvision.datasets.CIFAR100(
                './data', train=True, download=True)
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
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size, pin_memory=True, num_workers=2)
        device = get_default_device()
        self.trainloader = DeviceDataLoader(self.trainloader, device)
        self.testloader = DeviceDataLoader(self.testloader, device)


def get_default_device():
    """Pick GPU if available, else CPU"""
    # if torch.cuda.is_available():
    #     return torch.device('cuda')
    # else:
    #     return torch.device('cpu')
    return torch.device('mps')


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