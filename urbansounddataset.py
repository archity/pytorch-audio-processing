import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio


class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get the audio sample at 'index'
        audio_sample_path = self._get_audio_sample_path(index)

        # Get the label associated with this audio sample path
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

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
    ANNOTATIONS_FILE = "./datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "./datasets/UrbanSound8K/audio"
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
