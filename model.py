import torch
from torch import nn


handcrafted_feature_num = 34

class Model_CNN(nn.Module):
    def __init__(self, window_size, ori_feature_num, handcrafted_feature_num):
        super(Model_CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4, ori_feature_num), stride=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 1), stride=2),
            nn.Conv2d(in_channels=8, out_channels=14, kernel_size=(3, 1), stride=1),
            nn.BatchNorm2d(num_features=14),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 1), stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=14 * (window_size - 10) // 4, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    # Defining the forward pass
    def forward(self, inputs, handcrafted_feature):
        x = inputs.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

class Model_CNN_feature(nn.Module):
    def __init__(self, window_size, ori_feature_num, handcrafted_feature_num):
        super(Model_CNN_feature, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4, ori_feature_num), stride=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 1), stride=2),
            nn.Conv2d(in_channels=8, out_channels=14, kernel_size=(3, 1), stride=1),
            nn.BatchNorm2d(num_features=14),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(2, 1), stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=14 * (window_size - 10) // 4 + handcrafted_feature_num, out_features=1)
        )

    # Defining the forward pass
    def forward(self, inputs, handcrafted_feature):
        x = inputs.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        combined_x = torch.cat((x, handcrafted_feature), dim=1)
        out = self.linear(combined_x)
        return out

class Model_LSTM(nn.Module):
    def __init__(self, window_size, ori_feature_num, handcrafted_feature_num):
        super(Model_LSTM, self).__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=ori_feature_num, hidden_size=50, num_layers=1, dropout=0.1)
        self.linear = nn.Sequential(
            nn.Linear(in_features=50, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    # Defining the forward pass
    def forward(self, inputs, handcrafted_feature):
        x, (hn, cn) = self.lstm(inputs)
        out = self.linear(x[:, -1, :])
        return out

class Model_LSTM_feature(nn.Module):
    def __init__(self, window_size, ori_feature_num, handcrafted_feature_num):
        super(Model_LSTM_feature, self).__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=ori_feature_num, hidden_size=50, num_layers=1, dropout=0.1)
        self.linear = nn.Sequential(
            nn.Linear(in_features=50 + handcrafted_feature_num, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    # Defining the forward pass
    def forward(self, inputs, handcrafted_feature):
        x, (hn, cn) = self.lstm(inputs)
        combined_x = torch.cat((x[:, -1, :], handcrafted_feature), dim=1)
        out = self.linear(combined_x)
        return out

class Model(nn.Module):
    def __init__(self, window_size, ori_feature_num, handcrafted_feature_num):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=ori_feature_num, hidden_size=50, num_layers=1)
        self.attenion = Attention3dBlock()
        self.linear = nn.Sequential(
            nn.Linear(in_features=1500, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(inplace=True)
        )
        self.handcrafted = nn.Sequential(
            nn.Linear(in_features=handcrafted_feature_num, out_features=10),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=20, out_features=1)
        )

    def forward(self, inputs, handcrafted_feature):
        y = self.handcrafted(handcrafted_feature)
        x, (hn, cn) = self.lstm(inputs)
        x = self.attenion(x)
        # flatten
        x = x.reshape(-1, 1500)
        x = self.linear(x)
        out = torch.concat((x, y), dim=1)
        out = self.output(out)
        return out


class Attention3dBlock(nn.Module):
    def __init__(self):
        super(Attention3dBlock, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=30, out_features=30),
            nn.Softmax(dim=2),
        )

    # inputs: batch size * window size(time step) * lstm output dims
    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)
        x = self.linear(x)
        x_probs = x.permute(0, 2, 1)
        # print(torch.sum(x_probs.item()))
        output = x_probs * inputs
        return output
