import os

import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_file, transforms):
        self.data_dir = data_dir
        self._lf = labels_file
        self.labels = pd.read_csv(self._lf)
        self.transforms = transforms

    def __getitem__(self, index):
        filename = os.path.join(self.data_dir, self.labels.iloc[index, 0])
        data = self._read_data_sample(filename)
        label = self._get_label(index)
        sample = {'data': data, 'label': label}
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return self.labels.shape[0]

    def _read_data_sample(self, filename):
        return

    def _get_label(self, index):
        return

