import torch
from torch import nn


class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2, stride=1)  # 128
        self.batch1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1)  # 64
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1) # 32
        self.batch3 = nn.BatchNorm2d(32)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 64
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=28)

    def forward(self, image):
        out = self.conv1(image)
        out = self.batch1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.conv3(out)
        out = self.batch3(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.tanh(out)

        out = self.fc2(out)
        out = self.dropout(out)
        out = self.tanh(out)

        out = self.fc3(out)
        return out
