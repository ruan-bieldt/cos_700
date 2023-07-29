import sys
from data_handler import DataWrapper, get_default_device
from models import ResNet, ResNetMedium, ResNetLarge
from runner import ModelRunner


def main():
    run = sys.argv[1]
    dataloader = DataWrapper("cifar", 400)
    size = 1
    for i in range(10):
        model = ResNet(3, 100, size)
        model.to(get_default_device())
        runner = ModelRunner(model, dataloader.trainloader,
                             dataloader.testloader, 150, 0.001, 0.01, 0.01)
        runner.run(4)
        size += 2

    print("##############################################")
    print("Starting resnet medium run")
    model = ResNetMedium(3, 100, [2, 2, 2, 2])
    model.to(get_default_device())
    runner = ModelRunner(model, dataloader.trainloader,
                         dataloader.testloader, 200, 0.001, 0.01, 0.01)
    runner.run(4)
    print("##############################################")
    print("Starting resnet Large 1 run")
    model = ResNetLarge(3, 100, [3, 4, 6, 3])
    model.to(get_default_device())
    runner = ModelRunner(model, dataloader.trainloader,
                         dataloader.testloader, 200, 0.001, 0.01, 0.01)
    runner.run(4)
    print("##############################################")
    print("Starting resnet Large 2 run")
    model = ResNetLarge(3, 100, [3, 4, 23, 3])
    model.to(get_default_device())
    runner = ModelRunner(model, dataloader.trainloader,
                         dataloader.testloader, 200, 0.001, 0.01, 0.01)
    runner.run(4)
    print("##############################################")
    print("Starting resnet Large 3 run")
    model = ResNetLarge(3, 100, [3, 4, 36, 3])
    model.to(get_default_device())
    runner = ModelRunner(model, dataloader.trainloader,
                         dataloader.testloader, 200, 0.001, 0.01, 0.01)
    runner.run(4)


if __name__ == '__main__':
    main()
