import torch.nn as nn

import torch

num_classes = 11
class lstm2(nn.Module):
    def __init__(self):
        super(lstm2, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=2,
            hidden_size=128,
            num_layers=1,
            bias=False,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            bias=False,
            batch_first=True
        )

        if num_classes == 10:
            self.fc = nn.Linear(128*64, num_classes)
        if num_classes == 11:
            self.fc = nn.Linear(128*64, num_classes)
        if num_classes == 12:
            self.fc = nn.Linear(512*64, num_classes)

    def forward(self, x):

        x, _ = self.lstm1(x.transpose(2,1))
        x, _ = self.lstm2(x)
        x = torch.reshape(x, [x.shape[0],-1])
        x = self.fc(x)

        return x

# data = torch.randn(20,2,512)
# model = lstm2()
# print(model(data).shape)

# from torchinfo import summary
# model = lstm2().cuda()
# summary(model, input_size=(128, 2, 128))

