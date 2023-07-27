import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from components import conv_block, ResidualBlock, LargeResidualBlock


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes, n):
        super().__init__()

        self.conv1 = conv_block(in_channels, 32)
        self.conv2_x = self.make_layers(32, 32, n, False)
        self.conv3_x = self.make_layers(32, 64, n, True)
        self.conv4_x = self.make_layers(64, 128, n, True)

        self.classifier = nn.Linear(128, num_classes)

    def make_layers(self, channels_in, channels_out, n, downsample):
        layers = []
        if downsample:
            layers.append(ResidualBlock(channels_in, channels_out, 2))
            channels_in = channels_out
        else:
            layers.append(ResidualBlock(channels_in, channels_out))
        layers.append(ResidualBlock(channels_in, channels_out))
        for n in range(n-1):
            layers.append(ResidualBlock(channels_in, channels_out))
            layers.append(ResidualBlock(channels_in, channels_out))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = torch.mean(out, dim=[2, 3])
        out = self.classifier(out)

        return out


class ResNetLarge(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2_x = self.make_layers(64, 256, 3, False)
        self.conv3_x = self.make_layers(256, 512, 4, True)
        self.conv4_x = self.make_layers(512, 1024, 6, True)
        self.conv5_x = self.make_layers(1024, 2048, 3, True)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Linear(2048, num_classes))

    def make_layers(self, channels_in, channels_out, n, downsample):
        layers = []
        if downsample:
            layers.append(LargeResidualBlock(channels_in, channels_out, 2))
            channels_in = channels_out
        else:
            layers.append(LargeResidualBlock(channels_in, channels_out))
        layers.append(LargeResidualBlock(channels_in, channels_out))
        layers.append(LargeResidualBlock(channels_in, channels_out))
        for n in range(n-1):
            layers.append(LargeResidualBlock(channels_in, channels_out))
            layers.append(LargeResidualBlock(channels_in, channels_out))
            layers.append(LargeResidualBlock(channels_in, channels_out))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.classifier(out)

        return out
