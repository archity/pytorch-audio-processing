import torch

# Import the defined FF class and load function from train.py
from train import FeedForwardNet, download_mnist_datasets

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

def predict(model, input, target, class_mapping):
    model.eval()

    # Don't calculate any gradients, coz we are just evaluating, not training
    with torch.no_grad():

        # Predictions : Tensor(1, 10) -> [[0.1, 0.01, ..., 0.6]]
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)

        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

        return predicted, expected


if __name__ == "__main__":

    # Load back the model
    feedforward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feedforward_net.load_state_dict(state_dict=state_dict)

    # Load MNIST val dataset
    _, validation_data = download_mnist_datasets()

    # Get a sample from the validation dataset for inference
    input, target = validation_data[0][0], validation_data[0][1]

    # Make an inference
    predicted, expected = predict(feedforward_net, input, target, class_mapping)

    print(f"Predicted: '{predicted}', expected: '{expected}'")