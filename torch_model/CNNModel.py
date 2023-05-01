import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 32, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(32, 16, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.fc = nn.Linear(16 * 4, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 16 * 4)
        x = self.fc(x)
        return x


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 32, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(32, 16, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.fc = nn.Linear(16 * 4, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 16 * 4)
        x = self.fc(x)
        return x
