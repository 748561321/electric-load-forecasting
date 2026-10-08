import numpy as np
import pandas as pd
# 采用深度学习直接多步的策略 后续可以改为seq2seq
filepath = "../100_daysp2.csv"
data = pd.read_csv(filepath)
data.head()

data.shape

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(data[["data"]])
plt.xticks(range(0, data.shape[0], 10000), data["time"].loc[::10000], rotation=45)
plt.title("data", fontsize=18, fontweight="bold")
plt.xlabel("Time", fontsize=18)
plt.ylabel('Data', fontsize=18)
plt.show()

feature = data[["data"]]
feature.info()

# 区间压缩到 -1到1
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = MinMaxScaler()
feature["data"] = scaler.fit_transform(feature["data"].values.reshape(-1, 1))

feature.head(170000)
feature["data"].shape

def split_data(load, lookback):
    data_raw = load.to_numpy()
    print(data_raw.shape)
    data = []
    lookback +=24

    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index:index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-25]
    y_train = data[:train_set_size, -25:-1,0]

    x_test = data[train_set_size:, :-25]
    y_test = data[train_set_size:, -25:-1,0]

    return [x_train, y_train, x_test, y_test]

lookback = 24*7+1
x_train, y_train, x_test, y_test = split_data(feature, lookback)
print("x_train.shape = ", x_train.shape)
print("y_train.shape =", y_train.shape)
print("x_test.shape =", x_test.shape)
print("y_test.shape =", y_test.shape)
# exit()

import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_true = torch.from_numpy(y_train).type(torch.Tensor)
y_test_true = torch.from_numpy(y_test).type(torch.Tensor)


# 输入维度
input_dim = 1
# 隐藏层维度
hidden_dim = 128
# 循环的cell个数 即lstm层数
num_layers = 1
# 输出维度
output_dim = 24
# 循环
num_epoches = 1000


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)



criterion = torch.nn.MSELoss()  # 损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=0.005)  # 优化器

import time

train_loss_n = np.zeros(num_epoches)
test_loss_n = np.zeros(num_epoches)
start_time = time.time()
lstm = []

train_losses=[]
test_losses=[]
for epoch in range(num_epoches):
    # 训练
    model.train()
    # 将梯度清空
    optimiser.zero_grad()
    # 将数据放进去训练
    y_train_pred = model(x_train)
    # print(y_train_pred.shape)
    # print(y_train_true.shape)
    # 损失函数
    train_loss = criterion(y_train_pred, y_train_true)
    # 反向传播
    train_loss.backward()
    # 优化
    optimiser.step()

    print("Epoch ", epoch, "train_MSE: ", train_loss.item())
    train_loss_n[epoch] = train_loss.item()


    if epoch % 1==0:
        model.eval()
        with torch.no_grad():
            y_test_pred=model(x_test)
            test_loss=criterion(y_test_pred, y_test_true)
            test_loss_n[epoch] = test_loss.item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"epoch:{epoch},train_MSE:{train_loss},test_MSE:{test_loss}")

    torch.save(model, 'lstm.pth')

training_time = time.time() - start_time
print("Training time: {}".format(training_time))

# model = torch.load('lstm.pth')

def mse(pred_y, true_y):
    return np.mean((pred_y - true_y) ** 2)
def mae(pred_y,true_y):
    return np.mean(abs(pred_y-true_y))
def mape(pred_y, true_y):
    return np.mean(np.abs((pred_y-true_y) / true_y)) * 100

plt.title("Training Loss")
x = [i for i in range(num_epoches)]
plt.plot(x, train_loss_n, marker="o", markersize=1, label="pred_y")
plt.xlabel("Time")
plt.ylabel("Data")
plt.legend()
plt.savefig('../data/epoch_loss.png')
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
plt.savefig('../data/epoch_loss_last_half.png')
plt.show()

plt.figure()
y_train_pred = model(x_train).detach().numpy()
y_test_pred = model(x_test).detach().numpy()


pred_y = np.concatenate((y_train_pred, y_test_pred))
pred_y = scaler.inverse_transform(pred_y).T[0]
true_y = np.concatenate((y_train_true, y_test_true))
true_y = scaler.inverse_transform(true_y).T[0]

pred_train_y = scaler.inverse_transform(y_train_pred).T[0]
true_train_y = scaler.inverse_transform(y_train_true).T[0]
pred_test_y = scaler.inverse_transform(y_test_pred).T[0]
true_test_y = scaler.inverse_transform(y_test_true).T[0]
print(f"mae(pred_y,true_y):{mae(pred_train_y, true_train_y)}")
print(f"mape(pred_y,true_y):{mape(pred_train_y,true_train_y)}")
print(f"mae(pred_y,true_y):{mae(pred_test_y, true_test_y)}")
print(f"mape(pred_y,true_y):{mape(pred_test_y,true_test_y)}")
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
plt.savefig('../data/train_and_test.png')
plt.show()

import os

def create_file(filename):
    if os.path.exists(filename):
        # 如果文件已经存在，则更改文件名字
        filename = rename_file(filename)
    # 创建文件
    with open(filename, 'w') as f:
        f.write(f"filepath = {filepath}\n")
        f.write(f"input_dim = {input_dim}\n")
        f.write(f"hidden_dim = {hidden_dim}\n")
        f.write(f"num_layers = {num_layers}\n")
        f.write(f"output_dim = {output_dim}\n")
        f.write(f"num_epoches = {num_epoches}\n")
        f.write(f"lookback = {lookback}\n")
        for i in range(num_epoches):
            f.write(f"epoch:{i},train_MSE:{train_loss_n[i]},test_MSE:{test_loss_n[i]}\n")
        f.write("Training time: {}\n".format(training_time))
        f.write(f"mae(pred_y,true_y):{mae(pred_train_y, true_train_y)}")
        f.write(f"mape(pred_y,true_y):{mape(pred_train_y, true_train_y)}")
        f.write(f"mae(pred_y,true_y):{mae(pred_test_y, true_test_y)}")
        f.write(f"mape(pred_y,true_y):{mape(pred_test_y, true_test_y)}")
        f.write(f"mae(pred_y,true_y):{mae(pred_y, true_y)}\n")
        f.write(f"mape(pred_y,true_y):{mape(pred_y, true_y)}\n")

def rename_file(filename):
    filename, file_extension = os.path.splitext(filename)
    index = 1
    new_filename = f"{filename}_{index}{file_extension}"
    while os.path.exists(new_filename):
        index += 1
        new_filename = f"{filename}_{index}{file_extension}"
    return new_filename

# 示例用法
create_file("../data/train.txt")


import numpy as np


file_path = "../data.txt"
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"文件 {file_path} 已删除")

# 创建新文件
with open(file_path, 'w') as f:
    for val in true_y:
        f.write(str(val) + " ")

    f.write("\n")
    for val in pred_y:
        f.write(str(val) + " ")

    f.write("\n")
    for i in range(len(true_y)):
        f.write(str(np.abs(true_y[i] - pred_y[i])) + " ")


# 打开文件
with open("../data.txt", "r") as file:
    # 读取两行数据
    lines = file.readlines()[:3]

# 将每行数据拆分为字符串数组，并将其转换为浮点数
data = []
for line in lines:
    values = line.strip().split()  # 去除首尾空白字符并按空格分割
    float_values = [float(value) for value in values]  # 将字符串转换为浮点数
    data.append(float_values)

# 将数据转换为 NumPy 数组
data_array = np.array(data)

print(len(data_array[0]))
print(len(data_array[1]))
print(len(data_array[2]))
# 打印数据数组
print(data_array)
index = int(len(data_array[0])*0.2)
print(f"mae(pred_y,true_y):{int(mae(data_array[0,:-index], data_array[1,:-index]))}")
print(f"mape(pred_y,true_y):{mape(data_array[0,:-index], data_array[1,:-index]):.1f}")
print(f"mae(pred_y,true_y):{int(mae(data_array[0,-index:], data_array[1,-index:]))}")
print(f"mape(pred_y,true_y):{mape(data_array[0,-index:], data_array[1,-index:]):.1f}")
# print(f"mae(pred_y,true_y):{int(mae(data_array[0], data_array[1]))}")
# print(f"mape(pred_y,true_y):{mape(data_array[0], data_array[1]):.1f}")
max_abs_error = np.max(np.abs(data_array[0]-data_array[1]))
max_percentage_error = np.max(np.abs((data_array[0]- data_array[1]) / data_array[1])) * 100
print(f"max absolute error: {int(max_abs_error)}")
print(f"max percentage error: {max_percentage_error:.1f}%")

max_index = np.argmax(data_array[2])
print(data_array[0,max_index])
print(data_array[1,max_index])

