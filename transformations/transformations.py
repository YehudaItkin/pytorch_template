import torch
import cv2
import torchvision.transforms.functional as F
import numpy as np

__all__ = ['Resize',
           'RandomHorizontalFlip',
           'RandomVerticalFlip',
           'RandomCrop',
           'Normalize',
           'ToTensor'
           ]


class Resize:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        data = sample['data']
        labels = sample['labels']
        h, w = data.shape[:2]
        data = cv2.resize(data, (self.size, self.size), interpolation=self.interpolation)
        labels[:, 0] = (labels[:, 0] * self.size / w).astype('int')
        labels[:, 1] = (labels[:, 1] * self.size / h).astype('int')
        sample['data'] = data
        sample['labels'] = labels
        return sample


class RandomHorizontalFlip:
    def __init__(self, probability):
        self.prob = probability

    def __call__(self, sample):
        if np.random.sample() >= self.prob:
            return sample
        data = sample['data']
        labels = sample['labels']
        data = data[::-1, :]
        labels[:, 1] = data.shape[0] - labels[:, 1]
        sample['data'] = data
        sample['labels'] = labels
        return sample


class RandomVerticalFlip:
    def __init__(self, probability):
        self.prob = probability

    def __call__(self, sample):
        if np.random.sample() >= self.prob:
            return sample
        data = sample['data']
        labels = sample['labels']
        data = data[:, ::-1]
        labels[:, 0] = data.shape[1] - labels[:, 0]
        sample['data'] = data
        sample['labels'] = labels
        return sample


class RandomCrop(object):
    """Crop randomly the data in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, probability):
        self.prob = probability
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if np.random.sample() > self.prob:
            res = Resize(self.output_size[0])
            return res(sample)
        data, labels = sample['data'], sample['labels']

        h, w = data.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        data = data[top: top + new_h,
                left: left + new_w]

        labels = labels - [left, top]

        return {'data': data, 'labels': labels}


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['data'] = F.normalize(sample['data'], mean=self.mean, std=self.std)
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, labels = sample['data'], sample['labels']

        # swap color axis because
        # numpy data: H x W x C
        # torch data: C X H X W
        data = data.transpose((2, 0, 1)).copy()
        labels = labels.reshape(-1).copy()
        return {'data': torch.from_numpy(data).float(),
                'labels': torch.from_numpy(labels).float()}


