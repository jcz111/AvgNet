import torch.nn as nn
import torch
import torch.nn.functional as F

num_classes = 11
# ResNet {{{
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3), bias=False)
        self.drop2 = nn.Dropout(p=0.5)
        if num_classes == 11:
            self.dense = nn.Linear(10080, 256)
        elif num_classes == 12:
            self.dense = nn.Linear(40800, 256)
        elif num_classes == 10:
            self.dense = nn.Linear(10080, 256)
        self.drop3 = nn.Dropout(p=0.5)
        self.classfier = nn.Linear(256, num_classes)


    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x)).squeeze()
        x = self.drop2(x).view(x.size(0), -1)
        x = F.relu(self.dense(x))
        x = self.drop3(x)
        x = self.classfier(x)
        return x



        
def cnn2d(**kwargs):
    return CNN2D(**kwargs)
# data = torch.randn(10,2,512)
# model = cnn2d()
# out = model(data)
# print(out.shape)
# from torchsummary import summary
# model = cnn2d().cuda()
# summary(model, (2, 128))

# from torchinfo import summary
# model = cnn2d().cuda()
# summary(model, input_size=(128, 2, 128))

