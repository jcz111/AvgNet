import torch.nn as nn

import torch
import torch.nn.functional as F

num_classes = 11
class gru2(nn.Module):
    def __init__(self):
        super(gru2, self).__init__()

        self.gru1 = nn.GRU(
            input_size=2,
            hidden_size=128,
            num_layers=1,
            bias=False,
            batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            bias=False,
            batch_first=True
        )

        if num_classes == 10:
            self.fc1 = nn.Linear(128*64, 64)
            self.fc2 = nn.Linear(64, num_classes)
        if num_classes == 11:
            self.fc1 = nn.Linear(128*64, 64)
            self.fc2 = nn.Linear(64, num_classes)
        if num_classes == 12:
            self.fc1 = nn.Linear(512*64, 64)
            self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        x, _ = self.gru1(x.transpose(2,1))
        x = F.relu(x)
        x, _ = self.gru2(x)
        x = torch.reshape(x, [x.shape[0],-1])
        x = self.fc1(x)
        x = self.fc2(x)

        return x

