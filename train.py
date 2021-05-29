import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Add bunch of dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def download_mnist_datasets():
    """
    :return: The train and test data downloaded from torchvision's MNIST repository
    """
    train_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
    validation_data = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())

    return train_data, validation_data


if __name__  == "__main__":
    # Download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # Create data loader for train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # Build model

    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    feed_forward_net = FeedForwardNet().to(device=device)
