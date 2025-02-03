
import gym
import gym_donkeycar


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNController(nn.Module):
    def __init__(self):
        super(CNNController, self).__init__()
        # 输入通道数为3，输出通道数为16，卷积核大小为3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 全连接层
        self.fc1 = nn.Linear(64 * 20 * 15, 128)  # 展平后输入到全连接层
        self.fc2 = nn.Linear(128, 2)  # 输出2个分类

    def forward(self, x):
        # 卷积层 + ReLU 激活 + 池化层
        x = self.pool(F.relu(self.conv1(x)))  # 输出尺寸: (16, 80, 60)
        x = self.pool(F.relu(self.conv2(x)))  # 输出尺寸: (32, 40, 30)
        x = self.pool(F.relu(self.conv3(x)))  # 输出尺寸: (64, 20, 15)

        # 展平
        x = x.view(-1, 64 * 20 * 15)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出为2

        return x



if __name__ == '__main__':
    env = gym.make("donkey-generated-roads-v0")
    obs = env.reset()
    ctelist = []
    device='cuda'
    model = CNNController().to(device)
    model.load_state_dict(torch.load("model_epoch_10.pth"))
    for i in range(500):

        # env.pause_env()  # unfreeze
        img = obs
        # img = function_switch(img, noise)

        tensor = torch.from_numpy(img)

        # 调整维度，从 (120, 160, 3) 转换为 (3, 120, 160)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.float()
        tensor = tensor/255
        tensor = tensor.to(device)
        action = model(tensor.unsqueeze(0))

        result = action.squeeze().tolist()

        obs, reward, done, info = env.step([result[1],result[0]])
        ctelist.append(np.abs(info['cte']+2))
        print(info['cte'])

        print(ctelist)
    print('mean')
    print(np.mean(ctelist))

