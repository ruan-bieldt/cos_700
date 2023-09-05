from Models import *
import utils
from logger import SummaryLogger
import random
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
import argparse
import json
import time
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Pharaphaser training')
parser.add_argument('--text', default='log.txt', type=str)
parser.add_argument('--exp_name', default='cifar10/FT', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--lr', default='0.1', type=float)
parser.add_argument('--resume_epoch', default='0', type=int)
parser.add_argument('--epoch', default='163', type=int)
parser.add_argument('--decay_epoch', default=[82, 123], nargs="*", type=int)
parser.add_argument('--w_decay', default='5e-4', type=float)
parser.add_argument('--cu_num', default='0', type=str)
parser.add_argument('--seed', default='1', type=str)
parser.add_argument('--load_pretrained_teacher',
                    default='trained/Teacher.pth', type=str)
parser.add_argument('--load_pretrained_paraphraser',
                    default='trained/Paraphraser.pth', type=str)
parser.add_argument('--save_model', default='ckpt.t7', type=str)
parser.add_argument('--rate', type=float, default=0.5,
                    help='The paraphrase rate k')
parser.add_argument('--beta', type=int, default=500)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
print(args)


#### random Seed ####
num = random.randint(1, 10000)
random.seed(num)
torch.manual_seed(num)
#####################


os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num

# Data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

# Other parameters
DEVICE = torch.device("cuda")
RESUME_EPOCH = args.resume_epoch
DECAY_EPOCH = args.decay_epoch
DECAY_EPOCH = [ep - RESUME_EPOCH for ep in DECAY_EPOCH]
FINAL_EPOCH = args.epoch
EXPERIMENT_NAME = args.exp_name
W_DECAY = args.w_decay
base_lr = args.lr
RATE = args.rate
BETA = args.beta


# Load pretrained models
Teacher = ResNet56()
path = args.load_pretrained_teacher
state = torch.load(path, map_location=torch.device(DEVICE))
utils.load_checkpoint(Teacher, state)
Teacher.to(DEVICE)


# student models
Student = ResNet20()
Student.to(DEVICE)

# Loss and Optimizer
criterion_CE = nn.CrossEntropyLoss()
criterion = nn.L1Loss()

optimizer = optim.SGD(Student.parameters(), lr=base_lr,
                      momentum=0.9, weight_decay=W_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=DECAY_EPOCH, gamma=0.1)


def z_score_normalization(feature_maps):
    # Compute the mean and standard deviation along the batch and channel dimensions
    mean = torch.mean(feature_maps, dim=(0, 2, 3), keepdim=True)
    std = torch.std(feature_maps, dim=(0, 2, 3), keepdim=True)

    # Apply z-score normalization
    # Adding a small epsilon to avoid division by zero
    normalized_feature_maps = (feature_maps - mean) / (std + 1e-6)

    return normalized_feature_maps


def apply_svd(feature_maps, num_singular_values_to_keep):
    # Reshape the feature maps for SVD
    batch_size, num_channels, height, width = feature_maps.shape
    reshaped_feature_maps = feature_maps.view(batch_size, num_channels, -1)
    # Compute SVD
    U, S, V = torch.svd(reshaped_feature_maps)

    # Keep only the top 'num_singular_values_to_keep' singular values/components
    U = U[:, :, :num_singular_values_to_keep]
    S = S[:, :num_singular_values_to_keep]
    V = V[:, :, :num_singular_values_to_keep]

    # Reconstruct the feature maps
    reconstructed_feature_maps = torch.matmul(
        U, torch.matmul(torch.diag_embed(S), V.permute(0, 2, 1)))
    reconstructed_feature_maps = reconstructed_feature_maps.view(
        batch_size, num_channels, height, width)

    return reconstructed_feature_maps


def eval(net):
    loader = testloader
    flag = 'Test'

    epoch_start_time = time.time()
    net.eval()
    val_loss = 0

    correct = 0

    total = 0
    criterion_CE = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)

        loss = criterion_CE(outputs[3], targets)
        val_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('%s \t Time Taken: %.2f sec' %
          (flag, time.time() - epoch_start_time))
    print('Loss: %.3f | Acc net: %.3f%%' %
          (train_loss / (b_idx + 1), 100. * correct / total))
    return val_loss / (b_idx + 1),  correct / total


def train(teacher, student, epoch):
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)

    teacher.eval()
    student.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        # Knowledge transfer with SVD loss at the last layer
        ###################################################################################
        teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)

        teacher_features = apply_svd(teacher_outputs[2].cpu(), 10)
        student_features = apply_svd(student_outputs[2].cpu(), 10)

        loss = BETA * (criterion(utils.FT(student_features.detach()), utils.FT(
            teacher_features.detach()))) + criterion_CE(student_outputs[3], targets)
        ###################################################################################
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Train s1 \t Time Taken: %.2f sec' %
          (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc net: %.3f%%|' %
          (train_loss / (b_idx + 1), 100. * correct / total))
    return train_loss / (b_idx + 1), correct / total


if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d %H:%M')
    if int(args.log_time):
        folder_name = 'SVD_{}'.format(time_log)

    path = os.path.join(EXPERIMENT_NAME, folder_name)
    if not os.path.exists('ckpt/' + path):
        os.makedirs('ckpt/' + path)
    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)

    # Save argparse arguments as logging
    # Instantiate logger
    logger = SummaryLogger(path)

    for epoch in range(RESUME_EPOCH, FINAL_EPOCH+1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")

        ### Train ###
        train_loss, acc = train(Teacher, Student, epoch)
        scheduler.step()

        ### Evaluate  ###
        val_loss, test_acc = eval(Student)

        f.write('EPOCH {epoch} \t'
                'ACC_net : {acc_net:.4f} \t  \n'.format(
                    epoch=epoch, acc_net=test_acc)
                )
        f.close()

    utils.save_checkpoint({
        'epoch': epoch,
        'state_dict': Student.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, 'ckpt/' + path, filename='Model_{}.pth'.format(epoch))

    utils.save_checkpoint({
        'epoch': epoch,
        'state_dict': Student.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, 'ckpt/' + path, filename='Model_{}.pth'.format(epoch))
