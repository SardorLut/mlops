import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from DatasetLoader.ImageLoader import ImageLoader


def train_and_test_split(path):
    path_to_train = os.path.join(path)
    dataset = ImageFolder(path_to_train)
    train_data, test_data, train_label, test_label = (
        train_test_split(dataset.imgs, dataset.targets, test_size=0.2, random_state=42)
    )
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    train_dataset = ImageLoader(train_data, train_transform)
    test_dataset = ImageLoader(test_data, test_transform)
    return train_dataset, test_dataset


train_dataset, test_dataset = train_and_test_split(path="dataset/training_set/training_set/")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
