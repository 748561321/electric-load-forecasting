import torch
import torch.nn as nn
from torchviz import make_dot

# 定义模型
class FeatureLayer(nn.Module):
    def __init__(self, input_dim):
        super(FeatureLayer, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        weight_scores = torch.sigmoid(self.linear(x))
        weighted_input = x * weight_scores
        return weighted_input

class CNNLayer(nn.Module):
    def __init__(self, input_dim):
        super(CNNLayer, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1d(x.permute(0, 2, 1))
        out = self.pool(out)
        out = out.permute(0, 2, 1)
        return out

class ALSTMCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ALSTMCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.attention = FeatureLayer(input_dim)
        self.cnn_pool = CNNLayer(input_dim)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.attention(x)
        x = self.cnn_pool(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = self.linear(x)
        out = out+x
        out = self.fc(out[:, -1, :])
        return out

# 创建模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ALSTMCNN(input_dim=10, hidden_dim=20, num_layers=2, output_dim=1)

# 创建虚拟输入
x = torch.randn(1, 5, 10).to(device)

# 使用torchviz创建模型结构图
y = model(x)
make_dot(y, params=dict(model.named_parameters()))
