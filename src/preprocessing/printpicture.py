import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
plt.rcParams['font.sans-serif']=['SimHei']
# file_path = 'C:/Users/city/Desktop/Graduation project/data/weidusp2.csv'
file_path = 'D:/NJU_ALL/Graduation project/data/weidusp2.csv'
df = pd.read_csv(file_path,encoding='utf-8',low_memory=False)
# df= df.iloc[1:, :].copy()
# print(df)
df.set_index("date", inplace=True)
# start_date = '20130101 0:00:00'
# end_date = '20130331 0:00:00'
# df_subset = df.loc[start_date:end_date]
#
# # 绘制折线图
# plt.figure(figsize=(10, 6))
# plt.plot(df_subset.index, df_subset["charge"], marker='o', linestyle='-')
# plt.title('2024年1月1日到2024年3月31日的电力负荷变化')
# plt.xlabel('时间')
# plt.ylabel('电力负荷')
# plt.gca().xaxis.set_major_locator(plt.MaxNLocator(720))
# plt.show()

# ---------------------------------------------------------------------

# for i in range(7):
#     start_index = (i + 187) * 24
#     # 20120708 - 20120714
#     df_sub = df.iloc[start_index:start_index + 24]
#     print(df_sub.index)
#     x_values = [j for j in range(24)]
#     plt.plot(x_values, df_sub["charge"], marker='o', linestyle='-', label=f' 7.{i+8}')
#
#
# # 设置标题和标签
# # plt.title('2012年7月8日至7月14日电力负荷变化')
# plt.xlabel('time(h)', fontsize=14)
# plt.ylabel('power load(MW)', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
#
# plt.show()

# ---------------------------------------------------------------------

# for i in range(7):
#     start_index = (random.randint(0, 365)) * 24
#
#     df_sub = df.iloc[start_index:start_index + 24]
#     date = df_sub.index[0][4:8]
#     month =str(int(date[0:2]))
#     day = str(int(date[2:4]))
#     x_values = [j for j in range(24)]
#     plt.plot(x_values, df_sub["charge"], marker='o', linestyle='-', label=f' {month}.{day}')
#
#
# # 设置标题和标签
# # plt.title('2012年7月8日至7月14日电力负荷变化')
# plt.xlabel('time(h)', fontsize=14)
# plt.ylabel('Power Load(MW)', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# plt.show()
#

# ---------------------------------------------------------------------

# for i in range(4):
#     start_index = (i+ 5) * 24*7 - 24
#     # 20120206(周一)
#     df_sub = df.iloc[start_index:start_index + 24*7]
#     print(df_sub.index)
#     x_values = [j for j in range(24*7)]
#     plt.plot(x_values, df_sub["charge"],
#              # marker='o',
#              linestyle='-', label=f' 第{i+1}周')
#
#
# # 设置标题和标签
# plt.title('2012年')
# plt.xlabel('时间(h)', fontsize=14)
# plt.ylabel('电力负荷(MW)', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14,loc='upper right')
# plt.show()

# ---------------------------------------------------------------------
# for i in range(3):
#     start_index = (i) * 24*365
#     # 20120206(周一)
#     df_sub = df.iloc[start_index:start_index + 24*365:24]
#     print(df_sub.index)
#     x_values = [j for j in range(len(df_sub.index))]
#     plt.plot(x_values, df_sub["avg_charge"],linestyle='-', label=f' {2012+i}年')
#
#
#
# interval_size = len(df_sub.index)/12
#
# plt.yticks(fontsize=14)
# plt.xticks(
#     [i *interval_size for i in range(12)],
#     [j for j in range(1,13)],fontsize=14
# )
# plt.xlabel('时间(月)',fontsize = 14)
# plt.ylabel('电力负荷(MW)',fontsize = 14)
# plt.legend(fontsize = 14)
# plt.show()

# ---------------------------------------------------------------------

# for i in range(25):
#     start_index = (random.randint(0, 365)) * 24
#
#     holiday = df.iloc[start_index,-1]
#     col = "red"
#     while holiday == 0:
#         start_index = (random.randint(0, 365)) * 24
#         holiday = df.iloc[start_index, -1]
#
#     df_sub = df.iloc[start_index:start_index + 24]
#     date = df_sub.index[0][4:8]
#     month =str(int(date[0:2]))
#     day = str(int(date[2:4]))
#     x_values = [j for j in range(24)]
#
#     #     红色是节假日 蓝色是工作日 工作日在上面居多
#     plt.plot(x_values, df_sub["charge"],  linestyle='-', color = col,label='Red' if col == 'red' else 'Blue')
#
# for i in range(25):
#     start_index = (random.randint(0, 365)) * 24
#
#     holiday = df.iloc[start_index,-1]
#     col = "blue"
#     while holiday == 1:
#         start_index = (random.randint(0, 365)) * 24
#         holiday = df.iloc[start_index, -1]
#
#     df_sub = df.iloc[start_index:start_index + 24]
#     date = df_sub.index[0][4:8]
#     month =str(int(date[0:2]))
#     day = str(int(date[2:4]))
#     x_values = [j for j in range(24)]
#
#     #     红色是节假日 蓝色是工作日 工作日在上面居多
#     plt.plot(x_values, df_sub["charge"],  linestyle='-', color = col,label='Red' if col == 'red' else 'Blue')
#
#
# # 设置标题和标签
# # plt.title('2012年7月8日至7月14日电力负荷变化')
# plt.xlabel('time(h)',fontsize = 14)
# plt.ylabel('power load(MW)',fontsize = 14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# red_patch = plt.Line2D([0], [0], color='red', label='holiday')
# blue_patch = plt.Line2D([0], [0], color='blue', label='workday')
# plt.legend(handles=[red_patch, blue_patch],fontsize = 14)
# plt.show()

# ---------------------------------------------------------------------

# print(df["avg_charge"])
# print(df["avg_temp"])
#
# correlation1= np.corrcoef(df["avg_charge"],df["avg_temp"])
# correlation2= np.corrcoef(df["avg_charge"],df["humidity"])
# correlation3= np.corrcoef(df["avg_charge"],df["precipitation"])
#
#
#
# print("电力负荷和温度", correlation1[0, 1])
# print("电力负荷和相对湿度", correlation2[0, 1])
# print("电力负荷和降雨量", correlation3[0, 1])

# ---------------------------------------------------------------------

# for i in range(1):
#     start_index = (i) * 24*365
#     # 20120206(周一)
#     df_sub = df.iloc[start_index:start_index + 24*365:24]
#     print(df_sub)
#     x_values = [j for j in range(len(df_sub.index))]
#     plt.plot(x_values, df_sub["avg_charge"],linestyle='-', label=f' 日平均电荷')
#     # plt.plot(x_values, df_sub["avg_temp"],linestyle='-', label=f' 日平均温度')
#     # plt.plot(x_values, df_sub["humidity"],linestyle='-', label=f' 日相对湿度')
#     # plt.plot(x_values, df_sub["precipitation"],linestyle='-', label=f' 日降雨量')
#
# interval_size = len(df_sub.index)/12
#
# plt.xticks(
#     [i *interval_size for i in range(12)],
#     [j for j in range(1,13)],
#     fontsize =14
# )
# plt.yticks(fontsize =14)
# plt.xlabel('时间(月)',fontsize =14)
# plt.ylabel('电力负荷(MW)',fontsize =14)
# plt.legend(fontsize =14)
# plt.show()

# ---------------------------------------------------------------------



for i in range(3):
    start_index = (i) * 24*365
    # 20120206(周一)
    df_sub = df.iloc[start_index:start_index + 24*365:24]
    x_values = df_sub["avg_charge"]
    y_values = df_sub["avg_temp"]
    plt.scatter(x_values, y_values, label=f'{2012+i}')
plt.xticks(fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('power load(MW)',fontsize =14)
plt.ylabel('temperature(℃)',fontsize =14)
plt.legend(fontsize =14)
plt.show()
