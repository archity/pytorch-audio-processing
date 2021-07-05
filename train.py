import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "D:/Datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "D:/Datasets/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


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
    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Instantiate our dataset object
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    # Create data loader for train set
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)

    # Build model
    cnn = CNNNetwork().to(device)
    summary(model=cnn, input_size=(1, 64, 44))

    # Instantiate loss func + optimiser
    loss_func = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(model=cnn,
          data_loader=train_data_loader,
          loss_func=loss_func,
          optimiser=optimiser,
          device=device, epochs=EPOCHS)

    # Save the trained model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Model trained and saved to feedforwardnet.pth")
