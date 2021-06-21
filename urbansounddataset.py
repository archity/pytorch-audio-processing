import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torch


class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get the audio sample at 'index'
        audio_sample_path = self._get_audio_sample_path(index)

        # Get the label associated with this audio sample path
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        # Register the signal to the device
        signal = signal.to(self.device)

        # Make sure that sample rate is same for all
        signal = self._resample_if_necessary(signal, sr)

        # Use a single channel (mono) in case the audio has multi-channels
        signal = self._mix_down_if_necessary(signal)

        # In case our audio file has more samples than the ones we need (num_sammples)
        signal = self._cut_if_necessary(signal)

        # In case our audio file has less samples than the ones we need (num_sammples)
        signal = self._right_pad_if_necessary(signal)

        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (num_channels, num_samples) -> (1, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        # [1, 1, 1] -> [1, 1, 1, 0, 0]
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal

            # We want to do right-padding (append and NOT pre-pend)
            # Example [1, 1, 1] -> [1, 1, 1, 0, 0]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)

        # If audio is not mono (more than 1 channel)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        # Get the fold number in format "foldx", where x is the fold number
        # present in the 6th coloumn of the csv file.
        fold = f"fold{self.annotations.iloc[index, 5]}"

        # Get the complete path of the audio file
        # audio_dir/fold/{name of audio file}
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        # Get class label (7th coloumn in the CSV file)
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "D:/Datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "D:/Datasets/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[1]

