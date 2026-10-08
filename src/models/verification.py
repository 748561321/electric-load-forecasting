import numpy as np
def mae(pred_y,true_y):
    return np.mean(abs(pred_y-true_y))
def mape(pred_y, true_y):
    return np.mean(np.abs((pred_y-true_y) / true_y)) * 100

# pred_y = np.full((3, 3), 5)
# true_y = np.full((3, 3), 4.5)
#
# print(f"mae(pred_y,true_y):{int(mae(pred_y, true_y))}")
# print(f"mape(pred_y,true_y):{mape(pred_y, true_y):.1f}")


import numpy as np

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
print(f"mae(pred_y,true_y):{int(mae(data_array[0], data_array[1]))}")
print(f"mape(pred_y,true_y):{mape(data_array[0], data_array[1]):.1f}")
max_abs_error = np.max(np.abs(data_array[0]-data_array[1]))
max_percentage_error = np.max(np.abs((data_array[0]- data_array[1]) / data_array[1])) * 100
print(f"max absolute error: {int(max_abs_error)}")
print(f"max percentage error: {max_percentage_error:.1f}%")

max_index = np.argmax(data_array[2])
print(data_array[0,max_index])
print(data_array[1,max_index])

#  pred - true
train_pred_y = data_array[0,:-index]
train_true_y = data_array[1,:-index]
test_pred_y = data_array[0,-index:]
test_true_y = data_array[1,-index:]
true_y = list(train_true_y) + list(test_true_y)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))  # 设置图像大小为15x8英寸
plt.title("LSTM model")
x_train = [i for i in range(len(train_true_y))]
x_test = [i + len(train_true_y) for i in range(len(test_true_y))]
x_true = [i for i in range(len(true_y))]

# 合并的 true_y
plt.plot(x_true, true_y, marker="x", markersize=1, label="true_y", color="blue")

# 训练数据部分
plt.plot(x_train, train_pred_y, marker="o", markersize=1, label="train_pred_y", color="olive")

# 测试数据部分
plt.plot(x_test, test_pred_y, marker="o", markersize=1, label="test_pred_y", color="red")



plt.xlabel("Time(h)",fontsize=14)
plt.ylabel("Data(MW)",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
# save_path = os.path.join(folder_path, 'train_and_test.png')
# plt.savefig(save_path)
plt.show()