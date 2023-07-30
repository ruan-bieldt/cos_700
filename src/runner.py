import torch
import torch.nn as nn


class ModelRunner:
    def __init__(self, model, train_loader, test_loader, epochs, max_lr, grad_clip, weight_decay):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.opt_func = torch.optim.Adam
        self.epochs = epochs
        self.max_lr = max_lr
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.mac_accuracy = 0.0

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        outputs = [self.model.validation_step(
            batch) for batch in self.test_loader]
        return self.model.validation_epoch_end(outputs)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit_one_cycle(self, epochs):
        torch.cuda.empty_cache()
        history = []

        # Set up cutom optimizer with weight decay
        optimizer = self.opt_func(self.model.parameters(), self.max_lr,
                                  weight_decay=self.weight_decay)
        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.max_lr, epochs=epochs,
                                                    steps_per_epoch=len(self.train_loader))

        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            train_losses = []
            lrs = []
            for batch in self.train_loader:
                loss = self.model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # Gradient clipping
                if self.grad_clip:
                    nn.utils.clip_grad_value_(
                        self.model.parameters(), self.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(self.get_lr(optimizer))
                sched.step()

            # Validation phase
            result = self.evaluate()
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            self.model.epoch_end(epoch, result)
            history.append(result)
        return history

    def run(self, splits):
        print("Starting training\n")
        self.history = [self.evaluate()]
        print(self.history)
        for i in range(splits):
            print("Starting cycle " + str(i+1) + "\n")
            self.history += self.fit_one_cycle(int(self.epochs/splits))
            self.max_lr /= 10
