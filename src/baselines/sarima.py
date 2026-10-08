# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

time0 = '2012-01-01'
time1 = '2013-12-31'
col = '0:00:00'
forecast_steps = 30

energy = pd.read_csv('C:/Users/city/Desktop/Graduation project/data/数据统合单变量.csv')
energy['YMD'] = pd.to_datetime(energy['YMD'], format='%Y%m%d')
energy.set_index('YMD', inplace=True)
time_index = pd.date_range(time0, time1, freq='D')
ts = pd.Series(energy[col], index=time_index)

time2 = pd.to_datetime(time1) + pd.DateOffset(days=1)  # time1 之后一天开始
time3 = time2 + pd.DateOffset(days=forecast_steps-1)  # 测试集为30天
time_test = pd.date_range(time2, time3)
ts_test = pd.Series(energy[col], index=time_test)

# 绘制时间序列图和自相关、偏自相关图
plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(ts)
plt.title('Time Series Data')
plt.subplot(312)
plot_acf(ts, lags=40, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(313)
plot_pacf(ts, lags=40, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# 差分，使时间序列变得平稳
ts_diff = ts.diff().dropna()

# 绘制平稳化后的时间序列图和自相关、偏自相关图
plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(ts_diff)
plt.title('Differenced Time Series Data')
plt.subplot(312)
plot_acf(ts_diff, lags=40, ax=plt.gca())
plt.title('ACF after Differencing')
plt.subplot(313)
plot_pacf(ts_diff, lags=40, ax=plt.gca())
plt.title('PACF after Differencing')
plt.tight_layout()
plt.show()

# 拟合SARIMA模型
# model = auto_arima(ts, suppress_warnings=True, seasonal=True, m=90)  # 设置m为季节性的周期
# order = model.order
# seasonal_order = model.seasonal_order
order = (0,0,1)
seasonal_order = (0,2,2,9)
print(f"Selected SARIMA Order: {order}, Seasonal Order: {seasonal_order}")
model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
results = model.fit()

# 预测未来一段时间
forecast = results.get_forecast(steps=forecast_steps)

# 提取预测结果和置信区间
forecast_mean = forecast.predicted_mean
confidence_interval = forecast.conf_int()

mse = mean_squared_error(ts_test, forecast_mean)
mape = np.mean(np.abs((ts_test - forecast_mean) / ts_test)) * 100
print(ts_test)
print(forecast_mean)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# 绘制原始数据、预测结果和置信区间
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Observed')
plt.plot(forecast_mean, color='red', label='Forecast')
plt.fill_between(confidence_interval.index, confidence_interval.iloc[:, 0], confidence_interval.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
plt.title('SARIMA Forecast')
plt.legend()
plt.show()
