import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Add bunch of dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=256),
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


def train_one_epoch(model, data_loader, loss_func, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # Calculate loss
        predictions = model(input)
        loss = loss_func(predictions, target)

        # Reset gradients to zero after every batch of iteration
        optimiser.zero_grad()

        # Backpropogate loss and update weights
        loss.backward()  # Backpropogate
        optimiser.step()  # Update the weights

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_func, optimiser, device, epochs):
    """
    Function that trains over all the epochs, one by one.

    :param model: The feed-forward model class object
    :param data_loader: Pytorch's DataLoader class object, with defined batch size for loading
    :param loss_func: Function for evaluating th loss
    :param optimiser: Adam optimizer, with learning rate given as LEARNING_RATE
    :param device: CPU/GPU
    :param epochs: The number of EPOCHS defined
    :return:
    """
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_one_epoch(model, data_loader, loss_func, optimiser, device)
        print("----------------")
    print("Training finished")


if __name__ == "__main__":
    # Download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # Create data loader for train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Build model
    feed_forward_net = FeedForwardNet().to(device)

    # Instantiate loss func + optimiser
    loss_func = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    train(model=feed_forward_net,
          data_loader=train_data_loader,
          loss_func=loss_func,
          optimiser=optimiser,
          device=device, epochs=EPOCHS)

    # Save the trained model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and saved to feedforwardnet.pth")
