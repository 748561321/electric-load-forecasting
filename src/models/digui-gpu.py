import numpy as np
import pandas as pd

filepath = "../one_yearsp2.csv"
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
# feature["data"] = scaler.fit_transform(feature["data"].values.reshape(-1, 1))
feature.loc[:, "data"] = scaler.fit_transform(feature["data"].values.reshape(-1, 1))

# feature.head(170000)
# feature["data"].shape


def split_data(load, lookback,forward):
    data_raw = load.to_numpy()
    data = []

    for index in range(len(data_raw) - lookback -forward):
        data.append(data_raw[index:index + lookback + forward])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-forward-1]
    y_train = data[:train_set_size, -forward-1:-1, 0]

    x_test = data[train_set_size:, :-forward-1]
    y_test = data[train_set_size:, -forward-1:-1, 0]

    return [x_train, y_train, x_test, y_test]


# x_train.shape =  (41990, 95, 1)
# y_train.shape = (41990, 1)
# x_test.shape = (10498, 95, 1)
# y_test.shape = (10498, 1)

lookback = 24*3
forward = 24
x_train, y_train, x_test, y_test = split_data(feature, lookback,forward)
print("x_train.shape = ", x_train.shape)
print("y_train.shape =", y_train.shape)
print("x_test.shape =", x_test.shape)
print("y_test.shape =", y_test.shape)

# x_train.shape =  (1766, 167, 1)
# y_train.shape = (1766, 24)
# x_test.shape = (442, 167, 1)
# y_test.shape = (442, 24)

# x_train.shape =  (84403, 95, 1)
# y_train.shape = (84403, 1)
# x_test.shape = (21101, 95, 1)
# y_test.shape = (21101, 1)

# x_train.shape =  (6873, 167, 1)
# y_train.shape = (6873, 1)
# x_test.shape = (1718, 167, 1)
# y_test.shape = (1718, 1)
import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
# 真实的标签
y_train_true = torch.from_numpy(y_train).type(torch.Tensor)
y_test_true= torch.from_numpy(y_test).type(torch.Tensor)
# y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
# y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

# 输入维度
input_dim = 1
# 隐藏层维度
hidden_dim = 64
# 循环的cell个数 即lstm层数
num_layers = 1
# 输出维度
output_dim = 1
# 循环
num_epoches = 1000

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("CUDA1 is available.")
else:
    device = torch.device("cpu")
device = torch.device("cpu")

# 构建 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)    # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out



model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)



criterion = torch.nn.MSELoss()  # 损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器

model = model.to(device)
criterion = criterion.to(device)

x_train = x_train.to(device)
x_test = x_test.to(device)
y_train_true = y_train_true.to(device)
y_test_true = y_test_true.to(device)

import time

train_loss_n = np.zeros(num_epoches)
test_loss_n = np.zeros(num_epoches)
start_time = time.time()
lstm = []

train_losses=[]
test_losses=[]
# 修改训练循环
for epoch in range(num_epoches):
    # 训练
    model.train()
    # 将梯度清空
    optimiser.zero_grad()

    # 初始化输入和输出
    current_input = x_train  # 取第一个时间步的输入
    predicted_outputs = []

    for i in range(forward):
        # 将数据放进去训练
        temp = model(current_input)
        predicted_outputs.append(temp)  # 保存预测的值

        # 更新当前输入为预测值，准备下一个时间步的预测
        current_input = torch.cat([current_input[:, 1:, :], temp.unsqueeze(1)], dim=1)

    # 将所有预测的值合并成一个张量
    y_train_pred = torch.cat(predicted_outputs, dim=1)
    # print(y_train_pred[0,:])

    # 损失函数
    train_loss = criterion(y_train_pred, y_train_true)
    # 反向传播
    train_loss.backward()
    # 优化
    optimiser.step()

    # print("Epoch ", epoch, "train_MSE: ", train_loss.item())
    train_loss_n[epoch] = train_loss.item()

    if epoch % 1==0:
        model.eval()
        with torch.no_grad():
            # 递归预测测试集
            current_input = x_test # 取第一个时间步的输入
            predicted_outputs = []

            for i in range(forward):
                temp = model(current_input)
                predicted_outputs.append(temp)  # 保存预测的值

                # 更新当前输入为预测值，准备下一个时间步的预测
                current_input = torch.cat([current_input[:, 1:, :], temp.unsqueeze(1)], dim=1)

            # 将所有预测的值合并成一个张量
            y_test_pred = torch.cat(predicted_outputs, dim=1)
            # print(y_test_pred[0, :])
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
    return np.mean(np.abs((true_y - pred_y) / true_y)) * 100

import os
# def rename_file(filename):
#     filename, file_extension = os.path.splitext(filename)
#     index = 1
#     new_filename = f"{filename}_{index}{file_extension}"
#     while os.path.exists(new_filename):
#         index += 1
#         new_filename = f"{filename}_{index}{file_extension}"
#     return new_filename
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
x = [i for i in range(num_epoches)]
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

predicted_outputs = []
current_input = x_train
for i in range(forward):
    # 将数据放进去训练
    temp = model(current_input)
    predicted_outputs.append(temp)  # 保存预测的值

    # 更新当前输入为预测值，准备下一个时间步的预测
    current_input = torch.cat([current_input[:, 1:, :], temp.unsqueeze(1)], dim=1)
y_train_pred = torch.cat(predicted_outputs, dim=1)
y_train_pred = y_train_pred.cpu().detach().numpy()

predicted_outputs = []
current_input = x_test
for i in range(forward):
    # 将数据放进去训练
    temp = model(current_input)
    predicted_outputs.append(temp)  # 保存预测的值

    # 更新当前输入为预测值，准备下一个时间步的预测
    current_input = torch.cat([current_input[:, 1:, :], temp.unsqueeze(1)], dim=1)
y_test_pred = torch.cat(predicted_outputs, dim=1)
y_test_pred = y_test_pred.cpu().detach().numpy()
y_train_true = y_train_true.cpu().numpy()
y_test_true = y_test_true.cpu().numpy()

# print(type(y_train_pred))
# print(type(y_test_pred))
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
        f.write(f"num_epoches = {num_epoches}\n")
        f.write(f"lookback = {lookback}\n")
        for i in range(num_epoches):
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

        for val in true_y:
            f.write(str(val) + " ")

        f.write("\n\n\n\n")
        for val in pred_y:
            f.write(str(val) + " ")

        f.write("\n\n\n\n")
        for i in range(len(true_y)):
            f.write(str(np.abs(true_y[i] - pred_y[i])) + " ")



# 示例用法
save_path =os.path.join(folder_path, 'train.txt')
create_file(save_path)

print(f"mae(pred_y,true_y):{int(mae(pred_train_y, true_train_y))}")
print(f"mape(pred_y,true_y):{mape(pred_train_y, true_train_y):.1f}")
print(f"mae(pred_y,true_y):{int(mae(pred_test_y, true_test_y))}")
print(f"mape(pred_y,true_y):{mape(pred_test_y, true_test_y):.1f}")
def test(pred_y, true_y):
    return np.mean((np.abs(true_y - pred_y) / true_y)) * 100
print(f"test:{test(pred_test_y, true_test_y):.1f}")
print(f"max absolute error: {int(max_abs_error)}")
print(f"max percentage error: {max_percentage_error:.1f}%")

