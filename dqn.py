import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        ## TODO 1: network를 수정하세요.
        ## starts
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64*7*7, 784)
        self.fc2 = nn.Linear(784, 98)
        self.fc3 = nn.Linear(98, 2)

        ## end

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        ## TODO 2: 위에서 정의한 모델에 따라 forward 하는 코드를 작성하세요.
        ## starts
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        ## ends

        return x



