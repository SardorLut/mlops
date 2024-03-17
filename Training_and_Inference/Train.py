import torch
from tqdm import tqdm
from DatasetLoader.LoadDataset import train_loader
import torch.nn as nn
import torch.optim as optim
from .utils import init_model, device


class Train():
    def __init__(self, model):
        self.model = model

    def fit(self, train_loader, num_epoch, model, criterion, optimizer, device):
        for epoch in range(0, num_epoch):
            losses = []
            model.train()
            loop = tqdm(enumerate(train_loader), total=len(train_loader))  # progress bar
            for batch_idx, (data, targets) in loop:
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = model(data)

                loss = criterion(scores, targets)
                optimizer.zero_grad()
                losses.append(loss)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(scores, 1)
                loop.set_description(
                    f"Epoch {epoch + 1}/{num_epoch} process: {int((batch_idx / len(train_loader)) * 100)}")
                loop.set_postfix(loss=loss.data.item())

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'checpoint_epoch_' + str(epoch) + '.pt')


def train():
    model = init_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 1

    classifier = Train(model)
    classifier.fit(train_loader, num_epochs, model, criterion, optimizer, device)


if __name__ == '__main__':
    train()
