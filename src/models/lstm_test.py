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
feature["data"] = scaler.fit_transform(feature["data"].values.reshape(-1, 1))

feature.head(170000)
feature["data"].shape


def split_data(load, lookback):
    data_raw = load.to_numpy()
    data = []
    lookback +=24

    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index:index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-25]
    y_train = data[:train_set_size, -25:-1, 0]

    x_test = data[train_set_size:, :-25]
    y_test = data[train_set_size:, -25:-1, 0]

    return [x_train, y_train, x_test, y_test]


# x_train.shape =  (6854, 168, 1)
# y_train.shape = (6854, 24, 1)
# x_test.shape = (1713, 168, 1)
# y_test.shape = (1713, 24, 1)
lookback = 24*7+1
x_train, y_train, x_test, y_test = split_data(feature, lookback)
print("x_train.shape = ", x_train.shape)
print("y_train.shape =", y_train.shape)
print("x_test.shape =", x_test.shape)
print("y_test.shape =", y_test.shape)

import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
# 真实的标签
y_train_true = torch.from_numpy(y_train).type(torch.Tensor)
y_test_true = torch.from_numpy(y_test).type(torch.Tensor)
# y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
# y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

# 输入维度
input_dim = 1
# 隐藏层维度
hidden_dim = 64
# 循环的cell个数 即lstm层数
num_layers = 5
# 输出维度
output_dim = 1
# 循环
num_epoches = 200


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
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器

import time

hist = np.zeros(num_epoches)
start_time = time.time()
lstm = []

train_losses=[]
test_losses=[]
# for epoch in range(num_epoches):
#     # 训练
#     model.train()
#     # 将梯度清空
#     optimiser.zero_grad()
#     # 将数据放进去训练
#     y_train_pred = model(x_train)
#     # 损失函数
#     train_loss = criterion(y_train_pred, y_train_true)
#     # 反向传播
#     train_loss.backward()
#     # 优化
#     optimiser.step()
#
#     print("Epoch ", epoch, "train_MSE: ", train_loss.item())
#     hist[epoch] = train_loss.item()
#
#     if epoch % 1==0:
#         model.eval()
#         with torch.no_grad():
#             y_test_pred=model(x_test)
#             test_loss=criterion(y_test_pred,y_test_lstm)
#         train_losses.append(train_loss)
#         test_losses.append(test_loss)
#         print(f"epoch:{epoch},train_MSE:{train_loss},test_MSE:{test_loss}")
#
#     torch.save(model, 'lstm.pth')
# training_time = time.time() - start_time
# print("Training time: {}".format(training_time))

model = torch.load('oneyearduobu3000.pth')

def mse(pred_y, true_y):
    return np.mean((pred_y - true_y) ** 2)
def mae(pred_y,true_y):
    return np.mean(abs(pred_y-true_y))
def mape(pred_y, true_y):
    return np.mean(np.abs((pred_y-true_y) / true_y)) * 100


#
# plt.title("Training Loss")
# x = [i for i in range(num_epoches)]
# plt.plot(x, hist, marker="o", markersize=1, label="pred_y")
# plt.xlabel("Time")
# plt.ylabel("Data")
# plt.legend()
# plt.savefig('../data/epoch_loss.png')
# plt.show()
#
# plt.figure()
# # 看最后收敛程度
# half_index = len(hist) // 2
# x_half = x[half_index:]
# hist_half = hist[half_index:]
# plt.title("Training Loss (Last Half of Epochs)")
# plt.plot(x_half, hist_half, marker="o", markersize=1, label="pred_y")
# plt.xlabel("Time")
# plt.ylabel("Data")
# plt.legend()
# plt.savefig('../data/epoch_loss_last_half.png')
# plt.show()

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

plt.title("LSTM model")
x = [i for i in range(len(true_y))]
plt.plot(x, pred_y, marker="o", markersize=1, label="pred_y")
plt.plot(x, true_y, marker="x", markersize=1, label="true_y")
plt.xlabel("Time")
plt.ylabel("Data")
plt.legend()
plt.savefig('../data/train_and_test.png')
plt.show()





