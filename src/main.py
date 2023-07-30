import sys
from data_handler import DataWrapper, get_default_device
from models import ResNet, ResNetMedium, ResNetLarge
from runner import BaseTrainer


def main():
    run = sys.argv[1]
    mode = sys.argv[2]
    print("Loading data")
    print("##############################################")
    dataloader = DataWrapper("cifar", 400)
    size = 1
    # for i in range(10):
    #     print("Starting run with n=" + str(i))

    #     model = ResNet(3, 100, size)
    #     model.to(get_default_device())
    #     runner = BaseTrainer(model, dataloader.trainloader,
    #                          dataloader.testloader, 150, 0.001, 0.01, 0.01)
    #     runner.run(4)
    #     size += 2
    #     print("##############################################")
    if mode == "med":
        print("##############################################")
        print("Starting resnet medium 1 run")
        name = "resnet18_"+str(run)
        model = ResNetMedium(3, 100, [2, 2, 2, 2])
        model.to(get_default_device())
        runner = BaseTrainer(model, name, dataloader.trainloader,
                             dataloader.testloader, 400, 0.01, 0.01, 0.01)
        runner.run(5)
        print("##############################################")
        print("Starting resnet medium 2 run")
        name = "resnet34_"+str(run)
        model = ResNetMedium(3, 100, [3, 4, 6, 3])
        model.to(get_default_device())
        runner = BaseTrainer(model, name, dataloader.trainloader,
                             dataloader.testloader, 400, 0.01, 0.01, 0.01)
        runner.run(5)
    elif mode == "large":
        dataloader = DataWrapper("cifar", 100)
        print("##############################################")
        print("Starting resnet Large 1 run")
        name = "resnet50_"+str(run)
        model = ResNetLarge(3, 100, [3, 4, 6, 3])
        model.to(get_default_device())
        runner = BaseTrainer(model, name, dataloader.trainloader,
                             dataloader.testloader, 400, 0.01, 0.01, 0.01)
        runner.run(5)
        print("##############################################")
        print("Starting resnet Large 2 run")
        name = "resnet101_"+str(run)
        model = ResNetLarge(3, 100, [3, 4, 23, 3])
        model.to(get_default_device())
        runner = BaseTrainer(model, name, dataloader.trainloader,
                             dataloader.testloader, 400, 0.01, 0.01, 0.01)
        runner.run(5)
        print("##############################################")
        print("Starting resnet Large 3 run")
        name = "resnet152_"+str(run)
        model = ResNetLarge(3, 100, [3, 4, 36, 3])
        model.to(get_default_device())
        runner = BaseTrainer(model, name, dataloader.trainloader,
                             dataloader.testloader, 400, 0.01, 0.01, 0.01)
        runner.run(5)


if __name__ == '__main__':
    main()
