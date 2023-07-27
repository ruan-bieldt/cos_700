import sys
from data_handler import DataWrapper, get_default_device
from models import ResNet, ResNetLarge
from runner import ModelRunner


def main():
    run = sys.argv[1]
    dataloader = DataWrapper("cifar", 400)
    size = 1
    for i in range(5):
        model = ResNet(3, 100, size)
        model.to(get_default_device())
        size += 2
        runner = ModelRunner(model, dataloader.trainloader,
                             dataloader.testloader, 400, 0.001, 0.01, 0.01)
        runner.run(3)


if __name__ == '__main__':
    main()
