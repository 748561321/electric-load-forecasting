import numpy as np
import pandas as pd

def split_data(load, lookback,forward,interval =1):
    data_raw = load.to_numpy()
    data = []

    for index in range(len(data_raw) - lookback -forward):
        data.append(data_raw[index:index + lookback + forward])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-forward-1]
    y_train = data[:train_set_size, -forward-1+interval-1:-1:interval, 0]

    x_test = data[train_set_size:, :-forward-1]
    y_test = data[train_set_size:, -forward-1+interval-1:-1:interval, 0]

    return [x_train, y_train, x_test, y_test]

filepath = "../one_yearsp2.csv"
data = pd.read_csv(filepath)
data = data[["data"]]
# print(data.head(10))

print(data.shape)

x_train, y_train, x_test, y_test = split_data(data, 24*7,24,3)
print(x_train[0,:,:])
print(y_train[0,:])
# print(x_train[1,:,:])
# print(y_train[1,:])
print("x_train.shape = ", x_train.shape)
print("y_train.shape =", y_train.shape)
print("x_test.shape =", x_test.shape)
print("y_test.shape =", y_test.shape)

print("\n\n\n\n")

current_input = x_train[:, :1, :]  # 取第一个时间步的输入
predicted_outputs = []

# for i in range(24):  # 递归预测24个值
    # 将数据放进去训练
    # y_train_pred = model(current_input)
    # predicted_outputs.append(y_train_pred)  # 保存预测的值

    # 更新当前输入为预测值，准备下一个时间步的预测
    # current_input = torch.cat([current_input[:, 1:, :], y_train_pred.unsqueeze(1)], dim=1)