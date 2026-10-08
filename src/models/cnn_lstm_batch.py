import numpy as np
import pandas as pd

# mae(pred_y,true_y):195
# mape(pred_y,true_y):3.0
# mae(pred_y,true_y):219
# mape(pred_y,true_y):2.5


filepath = '../weidusp2.csv'
df = pd.read_csv(filepath, header=None, encoding='utf-8', low_memory=False, skiprows=0)
# print(df)
import matplotlib.pyplot as plt
import seaborn as sns

df_normalized = df.iloc[1:, 1:21].copy()
print(df_normalized)

from sklearn.preprocessing import MinMaxScaler

scaler_y = MinMaxScaler()
scaler_x = MinMaxScaler()
# 归一化同时将数据转换为float32
feature_y = scaler_y.fit_transform(df_normalized.iloc[:, 0:1])
feature_x = scaler_x.fit_transform(df_normalized.iloc[:, 1:])

feature = np.concatenate((feature_y, feature_x), axis=1)


# if (feature == MinMaxScaler().fit_transform(df_normalized)).any():
#     print("ok")

# feature = df_normalized

def split_data(load, lookback):
    data_raw = load
    data = []

    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index:index + lookback])

    data = np.array(data)
    print(data.shape)
    # (26208, 48, 20)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :, 1:]
    y_train = data[:train_set_size, 0:24, 0]

    x_test = data[train_set_size:, :, 1:]
    y_test = data[train_set_size:, 0:24, 0]

    return [x_train, y_train, x_test, y_test]


lookback = 24 * 2
x_train, y_train, x_test, y_test = split_data(feature, lookback)
print("x_train.shape = ", x_train.shape)
print("y_train.shape =", y_train.shape)
print("x_test.shape =", x_test.shape)
print("y_test.shape =", y_test.shape)
# print(x_train[1])
# print(y_train[1])
# x_train.shape =  (20966, 48, 19)
# y_train.shape = (20966, 48)
# x_test.shape = (5242, 48, 19)
# y_test.shape = (5242, 48)

import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_true = torch.from_numpy(y_train).type(torch.Tensor)
y_test_true = torch.from_numpy(y_test).type(torch.Tensor)

if torch.cuda.is_available():
    device = torch.device("cuda:3")
    print("CUDA is available.")
else:
    device = torch.device("cpu")


# device = torch.device("cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

# 权重提取
# class FeatureLayer(nn.Module):
#     # 输入 bsi 按照i进行算
#     def __init__(self):
#         super(FeatureLayer, self).__init__()
#     def forward(self, x):
#         seq_len, size = x.shape[1], x.shape[2]
#         w = nn.Linear(size,size)
#         w = w.to(device)
#         e = torch.matmul(w(x),x.permute(0,2,1))
#         attention_scores = F.softmax(e,dim = -1)
#         # weighted_input = x * attention_scores
#         out = torch.bmm(attention_scores, x)
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义权重层
class FeatureLayer(nn.Module):
    def __init__(self, input_dim):
        super(FeatureLayer, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)  # 输入维度为特征维度，输出维度与输入维度相同

    def forward(self, x):
        weight_scores = torch.sigmoid(self.linear(x))
        weighted_input = x * weight_scores
        return weighted_input

class FeatureAttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(FeatureAttentionLayer, self).__init__()
        self.w = nn.Linear(input_size, input_size)

    def forward(self, x):
        e = torch.matmul(self.w(x), x.permute(0, 2, 1))
        attention_scores = F.softmax(e, dim=-1)
        out = torch.bmm(attention_scores, x)
        out =F.relu(out)
        return out

class TimeAttentionLayer(nn.Module):
    def __init__(self, seq_len):
        super(TimeAttentionLayer, self).__init__()
        self.w = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        x = x.permute(0,2,1)
        e = torch.matmul(self.w(x), x.permute(0, 2, 1))
        attention_scores = F.softmax(e, dim=-1)
        out = torch.bmm(attention_scores, x)
        out = F.relu(out.permute(0, 2, 1))
        return out

class CNNLayer(nn.Module):
    def __init__(self, input_dim):
        super(CNNLayer, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)  # 假设输入和输出维度相同
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1d(x.permute(0, 2, 1))  # 将输入转换为(C_in, W_in, D_in)格式
        out = self.pool(out)
        out = out.permute(0, 2, 1)  # 将输出转换回(C_out, D_out, W_out)格式
        return out

# 更新LSTM模型
class ALSTMCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ALSTMCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.attention = FeatureLayer(input_dim)
        # self.attention2 = FeatureAttentionLayer(input_dim)
        # self.attention2 = TimeAttentionLayer(48//2)
        # CNN和池化层
        self.cnn_pool = CNNLayer(input_dim)

        self.linear = nn.Linear(input_dim, hidden_dim)

        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = self.attention(x)

        # 应用CNN和池化层
        x = self.cnn_pool(x)

        # print(x.shape)

        # x = self.attention2(x)

        # LSTM层
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        x = self.linear(x)
        out = out+x

        out = self.fc(out[:, -1, :])

        return out

# 定义输入维度等参数
input_dim = 19
hidden_dim = 128
num_layers = 1
output_dim = 24
num_epochs = 2000
# 创建模型
model = ALSTMCNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                 num_layers=num_layers)

criterion = torch.nn.MSELoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 优化器

model = model.to(device)
criterion = criterion.to(device)

x_train = x_train.to(device)
x_test = x_test.to(device)
y_train_true = y_train_true.to(device)
y_test_true = y_test_true.to(device)

import time

train_loss_n = np.zeros(num_epochs)
test_loss_n = np.zeros(num_epochs)
start_time = time.time()



from torch.utils.data import DataLoader, TensorDataset

# 构造 TensorDataset
train_dataset = TensorDataset(x_train, y_train_true)
test_dataset = TensorDataset(x_test, y_test_true)

# 定义批量大小
batch_size = 64

# 创建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失列表
train_losses = []
test_losses = []

# 训练循环
for epoch in range(num_epochs):
    # 训练模型
    model.train()
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_loss_n[epoch] = loss.item()

    # 测试模型
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())
            test_loss_n[epoch] = loss.item()

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Train Loss: {np.mean(train_losses)}, Test Loss: {np.mean(test_losses)}")

# 保存模型
torch.save(model.state_dict(), 'lstm.pth')
training_time = time.time() - start_time
print("Training time: {}".format(training_time))


# model = torch.load('lstm.pth')

def mse(pred_y, true_y):
    return np.mean((pred_y - true_y) ** 2)


def mae(pred_y, true_y):
    return np.mean(abs(pred_y - true_y))


def mape(pred_y, true_y):
    return np.mean(np.abs((pred_y - true_y) / true_y)) * 100


import os


def rename_folder(folder_path):
    index = 1
    new_folder_path = folder_path
    while os.path.exists(new_folder_path):
        new_folder_path = f"{folder_path}_{index}"
        index += 1
    return new_folder_path


folder_path = '../data/train'
folder_path = rename_folder(folder_path)
os.makedirs(folder_path)

plt.title("Training Loss")
x = [i for i in range(num_epochs)]
plt.plot(x, train_loss_n, marker="o", markersize=1, label="pred_y")
plt.xlabel("Time")
plt.ylabel("Data")
plt.legend()
save_path = os.path.join(folder_path, 'epoch_loss.png')
plt.savefig(save_path)
plt.show()

plt.figure()
# 看最后收敛程度
half_index = len(train_loss_n) // 2
x_half = x[half_index:]
hist_half = train_loss_n[half_index:]
plt.title("Training Loss (Last Half of Epochs)")
plt.plot(x_half, hist_half, marker="o", markersize=1, label="pred_y")
plt.xlabel("Time")
plt.ylabel("Data")
plt.legend()
save_path = os.path.join(folder_path, 'epoch_loss_last_half.png')
plt.savefig(save_path)
plt.show()

plt.figure()
y_train_pred = model(x_train).cpu().detach().numpy()
y_test_pred = model(x_test).cpu().detach().numpy()
y_train_true = y_train_true.cpu().numpy()
y_test_true = y_test_true.cpu().numpy()

pred_y = np.concatenate((y_train_pred, y_test_pred))
pred_y = scaler_y.inverse_transform(pred_y).T[0]
true_y = np.concatenate((y_train_true, y_test_true))
true_y = scaler_y.inverse_transform(true_y).T[0]

pred_train_y = scaler_y.inverse_transform(y_train_pred).T[0]
true_train_y = scaler_y.inverse_transform(y_train_true).T[0]
pred_test_y = scaler_y.inverse_transform(y_test_pred).T[0]
true_test_y = scaler_y.inverse_transform(y_test_true).T[0]

# if np.array_equal(true_y, np.concatenate((true_train_y, true_test_y))):
#     print("ok")

print(f"mae(pred_y,true_y):{mae(pred_train_y, true_train_y)}")
print(f"mape(pred_y,true_y):{mape(pred_train_y, true_train_y)}")
print(f"mae(pred_y,true_y):{mae(pred_test_y, true_test_y)}")
print(f"mape(pred_y,true_y):{mape(pred_test_y, true_test_y)}")
print(f"mae(pred_y,true_y):{mae(pred_y, true_y)}")
print(f"mape(pred_y,true_y):{mape(pred_y, true_y)}")

max_abs_error = np.max(np.abs(pred_y - true_y))
max_percentage_error = np.max(np.abs((pred_y - true_y) / true_y)) * 100
print(f"max absolute error: {max_abs_error}")
print(f"max percentage error: {max_percentage_error}%")

plt.title("LSTM model")
x = [i for i in range(len(true_y))]
plt.plot(x, pred_y, marker="o", markersize=1, label="pred_y")
plt.plot(x, true_y, marker="x", markersize=1, label="true_y")
plt.xlabel("Time(h)")
plt.ylabel("Data(MW)")
plt.legend()
save_path = os.path.join(folder_path, 'train_and_test.png')
plt.savefig(save_path)
plt.show()


def create_file(filename):
    # if os.path.exists(filename):
    #     # 如果文件已经存在，则更改文件名字
    #     filename = rename_file(filename)
    # 创建文件
    with open(filename, 'w') as f:
        script_path = os.path.realpath(__file__)
        f.write("当前执行程序的文件路径: {}\n".format(script_path))
        f.write(f"filepath = {filepath}\n")
        f.write(f"input_dim = {input_dim}\n")
        f.write(f"hidden_dim = {hidden_dim}\n")
        f.write(f"num_layers = {num_layers}\n")
        f.write(f"output_dim = {output_dim}\n")
        f.write(f"num_epoches = {num_epochs}\n")
        f.write(f"lookback = {lookback}\n")
        for i in range(num_epochs):
            f.write(f"epoch:{i},train_MSE:{train_loss_n[i]},test_MSE:{test_loss_n[i]}\n")
        f.write("Training time: {}\n".format(training_time))
        f.write(f"mae(pred_y,true_y):{mae(pred_train_y, true_train_y)}\n")
        f.write(f"mape(pred_y,true_y):{mape(pred_train_y, true_train_y)}\n")
        f.write(f"mae(pred_y,true_y):{mae(pred_test_y, true_test_y)}\n")
        f.write(f"mape(pred_y,true_y):{mape(pred_test_y, true_test_y)}\n")
        f.write(f"mae(pred_y,true_y):{mae(pred_y, true_y)}\n")
        f.write(f"mape(pred_y,true_y):{mape(pred_y, true_y)}\n")
        f.write(f"max absolute error: {max_abs_error}\n")
        f.write(f"max percentage error: {max_percentage_error}%\n")

        for val in pred_y:
            f.write(str(val) + " ")

        f.write("\n\n\n\n")
        for val in true_y:
            f.write(str(val) + " ")

        f.write("\n\n\n\n")
        for i in range(len(true_y)):
            f.write(str(np.abs(true_y[i] - pred_y[i])) + " ")

        f.write("\n")


save_path = os.path.join(folder_path, 'train.txt')
create_file(save_path)

print(f"mae(pred_y,true_y):{int(mae(pred_train_y, true_train_y))}")
print(f"mape(pred_y,true_y):{mape(pred_train_y, true_train_y):.1f}")
print(f"mae(pred_y,true_y):{int(mae(pred_test_y, true_test_y))}")
print(f"mape(pred_y,true_y):{mape(pred_test_y, true_test_y):.1f}")
print(f"max absolute error: {int(max_abs_error)}")
print(f"max percentage error: {max_percentage_error:.1f}%")