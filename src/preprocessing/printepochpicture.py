import numpy as np
import matplotlib.pyplot as plt
train_MSE_values = []
test_MSE_values = []
# 打开文件
with open("../epoch2.txt", "r") as file:
    # 读取两行数据
    for line in file:
        parts = line.strip().split(',')
        train_MSE = float(parts[1].split(':')[1])
        test_MSE = float(parts[2].split(':')[1])
        train_MSE_values.append(train_MSE)
        test_MSE_values.append(test_MSE)

epochs = range(len(train_MSE_values))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_MSE_values, marker='o',markersize=1, color='b')
plt.xticks(range(0, len(epochs)+1, len(epochs)//5),fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('Train MSE',fontsize = 14)
plt.xlabel('Epoch',fontsize = 14)
plt.ylabel('MSE',fontsize = 14)

plt.subplot(1, 2, 2)
plt.plot(epochs, test_MSE_values, marker='o',markersize=1, color='r')
plt.xticks(range(0, len(epochs)+1, len(epochs)//5),fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('Test MSE',fontsize = 14)
plt.xlabel('Epoch',fontsize = 14)
plt.ylabel('MSE',fontsize = 14)

plt.tight_layout()
plt.show()

epochs = range(int(len(train_MSE_values) * 0.8), len(train_MSE_values))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_MSE_values[int(len(train_MSE_values) * 0.8):], marker='o', markersize=1, color='b')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('Train MSE (Second Half)',fontsize =14)
plt.xlabel('Epoch',fontsize =14)
plt.ylabel('MSE',fontsize =14)

plt.subplot(1, 2, 2)
plt.plot(epochs, test_MSE_values[int(len(train_MSE_values) * 0.8):], marker='o',markersize=1,  color='r')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('Test MSE (Second Half)',fontsize =14)
plt.xlabel('Epoch',fontsize =14)
plt.ylabel('MSE',fontsize =14)

plt.tight_layout()
plt.show()