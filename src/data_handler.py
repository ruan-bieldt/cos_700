import torchvision
import numpy as np
import torch
import torchvision.transforms as tt
import imageio
from collections import defaultdict
from tqdm.autonotebook import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                               root_dir)
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
                 transform=None, download=False, max_samples=None):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(
                self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                     dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            img = _add_channels(img)
            lbl = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            img = _add_channels(img)
            lbl = None if self.mode == 'test' else s[self.label_idx]
        sample = {'image': img, 'label': lbl}

        if self.transform:
            sample = self.transform(sample)
        return sample


class DataWrapper:
    def __init__(self, name, batch_size):
        if name == "cifar":
            train_data = torchvision.datasets.CIFAR100(
                './data', train=True, download=True)
        elif name == "tiny":
            train_data = TinyImageNetDataset(
                '/mnt/lustre/users/rbieldt/cos_700/src/data/tiny-imagenet-200')
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
            trainset = TinyImageNetDataset("/mnt/lustre/users/rbieldt/cos_700/src/data/tiny-imagenet-200",
                                           train=True,
                                           transform=transform_train)
            testset = TinyImageNetDataset("/mnt/lustre/users/rbieldt/cos_700/src/data/tiny-imagenet-200",
                                          mode='test',
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
