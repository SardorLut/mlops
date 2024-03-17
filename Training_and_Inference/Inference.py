import torch
from .utils import init_model, device
import torch.nn as nn
import torch.optim as optim
from DatasetLoader.LoadDataset import test_loader


def inference(name):
    checkpoint_name = str(name)
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_name)
    else:
        checkpoint = torch.load(checkpoint_name, map_location='cpu')

    model = init_model()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    criterion = nn.CrossEntropyLoss()

    test(model, test_loader, criterion)


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, predictions = torch.max(output, 1)
            correct += (predictions == y).sum().item()
            test_loss = criterion(output, y)

    test_loss /= len(test_loader.dataset)
    print("Average Loss: ", test_loss, "  Accuracy: ", correct, " / ",
          len(test_loader.dataset), "  ", int(correct / len(test_loader.dataset) * 100), "%")


if __name__ == "__main__":
    inference("checpoint_epoch_0.pt")
