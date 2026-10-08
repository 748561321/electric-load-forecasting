import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math

from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
# from common.utils import load_data, mape
from IPython.display import Image

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller  # adf检验库
from statsmodels.stats.diagnostic import acorr_ljungbox  # 随机性检验库
from statsmodels.tsa.arima_model import ARMA

# %matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")  # specify to ignore warning messages

energy = pd.read_csv('C:/Users/city/Desktop/Graduation project/data/test.csv')
energy['time'] = pd.to_datetime(energy['time'])

energy.plot(y='data',x='time', subplots=True,
            figsize=(15, 8), fontsize=12)
plt.xlabel('time', fontsize=12)
plt.ylabel('data', fontsize=12)
# plt.show()

train_start_dt = '2014-12-01 00:00:00'
test_start_dt = '2014-12-30 00:00:00'
energy.set_index('time', inplace=True)
# print(energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)])

energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['data']].rename(columns={'data':'train'}) \
    .join(energy[test_start_dt:][['data']].rename(columns={'data':'test'}), how='outer') \
    .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('data', fontsize=12)
# plt.show()

train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['data']]
test = energy.copy()[energy.index >= test_start_dt][['data']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

# 归一化 这里暂且只保留两位小数
scaler = MinMaxScaler()
train['data'] = scaler.fit_transform(train)
# print(train.head(10).round(4))
test['data'] = scaler.transform(test)

# # 平稳性检验
# def adf_val(ts, ts_title):
#     '''
#     ts: 时间序列数据，Series类型
#     ts_title: 时间序列图的标题名称，字符串
#     '''
#     # 稳定性（ADF）检验
#     adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(ts)
#
#     name = ['adf', 'pvalue', 'usedlag',
#             'nobs', 'critical_values', 'icbest']
#     values = [adf, pvalue, usedlag, nobs,
#               critical_values, icbest]
#     print(list(zip(name, values)))
#
#     return adf, pvalue, critical_values,
#     # 返回adf值、adf的p值、三种状态的检验值
#
# ts_data = energy['data'].astype('float32')
# adf, pvalue1, critical_values = adf_val(
#           ts_data, 'raw time series')
# # [('adf', -9.305756466872985), ('pvalue', 1.0819482969678154e-15), ('usedlag', 69), ('nobs', 105530),
# # ('critical_values', {'1%': -3.4304119677728693, '5%': -2.861567388800577, '10%': -2.566784578097394}),
# # ('icbest', 1328966.155099571)]
# # 1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设。
# # 另外，P-value是否非常接近0，接近0，则是平稳的，否则，不平稳。

# # 白噪声检验
# ts_data = energy['data'].astype('float32')
# lbvalue, pvalue = acorr_ljungbox(ts_data, lags=1)
# print('Ljung-Box Statistic:', lbvalue)
# print('p-value:', pvalue)
# # 若p值远小于0.01，因此我们拒绝原假设，认为该时间序列是平稳的。（这里原假设是存在单位根，即时间序列为非平稳的。)


# # 确定阶数
# fig = plt.figure(figsize=(20,16))
# ax1= fig.add_subplot(211)
# diff1 = train.diff(1)
# diff1.plot(ax=ax1)
# ax2= fig.add_subplot(212)
# diff2 = train.diff(2)
# diff2.plot(ax=ax2)
# plt.show()

# # 模型选择
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# fig, axes = plt.subplots(2, 1, figsize=(12, 4*2))
# # 自相关
# plot_acf(train['data'],lags=12,title='raw_acf', ax=axes[0])
# # 偏自相关
# plot_pacf(train['data'],lags=12,title='raw_pacf', ax=axes[1])
# plt.show()
# # 其中lags 表示滞后的阶数，以上分别得到acf 图和pacf 图。

# 指定要提前预测的步骤数
# HORIZON = 3
# order = (1, 1, 1)
# seasonal_order = (1, 1, 0, 24)
# model = SARIMAX(endog=train, order=order,
#                 seasonal_order=seasonal_order)
# results = model.fit()
# print(results.summary())

from pmdarima import auto_arima
model = auto_arima(train, seasonal=True, m=96)
print("best model：", model) #拟合出来最佳模型

n_periods = 576;
y_pred = model.predict(n_periods) #预测未来n_periods期
mae = mean_absolute_error(test, y_pred)
print("mean_absolute_error： ", mae)
# results = model.fit()
# print(results.summary())