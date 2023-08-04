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
    def __init__(self, in_channels, num_classes, n, init_channels):
        super().__init__()

        self.conv1 = conv_block(in_channels, init_channels)
        self.conv2_x = self.make_layers(init_channels, init_channels, n, False)
        init_channels *= 2
        self.conv3_x = self.make_layers(
            init_channels//2, init_channels, n, True)
        init_channels *= 2
        self.conv4_x = self.make_layers(
            init_channels//2, init_channels, n, True)

        self.classifier = nn.Linear(init_channels, num_classes)

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


class ResNetMedium(ImageClassificationBase):
    def __init__(self, in_channels, num_classes, layers):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2_x = self.make_layers(64, 64, layers[0], False)
        self.conv3_x = self.make_layers(64, 128, layers[1], True)
        self.conv4_x = self.make_layers(128, 256, layers[2], True)
        self.conv5_x = self.make_layers(256, 512, layers[3], True)
        self.classifier = nn.Linear(512, num_classes)

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
        out = self.conv5_x(out)
        out = torch.mean(out, dim=[2, 3])
        out = self.classifier(out)

        return out


class ResNetLarge(ImageClassificationBase):
    def __init__(self, in_channels, num_classes, layers):
        super().__init__()

        self.conv1 = conv_block(in_channels, 128)
        self.conv2_x = self.make_layers(64, 256, layers[0], False)
        self.conv3_x = self.make_layers(128, 512, layers[1], True)
        self.conv4_x = self.make_layers(256, 1024, layers[2], True)
        self.conv5_x = self.make_layers(512, 2048, layers[3], True)
        self.classifier = nn.Linear(2048, num_classes)

    def make_layers(self, channels_in, channels_out, n, downsample):
        layers = []
        if downsample:
            layers.append(LargeResidualBlock(
                channels_in, channels_out, True, 2))
            channels_in = channels_out
        else:
            layers.append(LargeResidualBlock(channels_in, channels_out, True))
        for n in range(n-1):
            layers.append(LargeResidualBlock(channels_in, channels_out))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = torch.mean(out, dim=[2, 3])
        out = self.classifier(out)

        return out
