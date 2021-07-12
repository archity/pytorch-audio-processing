import torch
import torchaudio

from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
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
    cnn = CNNNetwork()
    state_dict = torch.load("trained_model.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict=state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device="cpu")

    # Get a sample from the validation dataset for inference
    # [num_channels, freq, time]
    index = 0
    input, target = usd[index][0], usd[index][1]

    # Introduce another dimension on the first (0th) index
    # [batchsize(=1), num_channels, freq, time]
    input.unsqueeze_(0)

    # Make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f"Predicted: '{predicted}', expected: '{expected}'")