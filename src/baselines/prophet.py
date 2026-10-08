import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
# 读取数据
energy = pd.read_csv('C:/Users/city/Desktop/Graduation project/data/数据统合单变量.csv')

# 设置时间范围
time0 = '2012-01-01'
time1 = '2013-12-31'
col = '0:00:00'
forecast_steps = 30

# 创建时间序列
energy['YMD'] = pd.to_datetime(energy['YMD'], format='%Y%m%d')
energy.set_index('YMD', inplace=True)
time_index = pd.date_range(time0, time1, freq='D')
# ts = pd.Series(energy[col], index=time_index)

# ts = pd.DataFrame({'ds': time_index, 'y': energy[col][time_index]}).set_index('ds')
ts = pd.DataFrame({'ds': pd.to_datetime(time_index), 'y': energy[col][time_index]})

# print(ts.head())

# 拟合Prophet模型
# model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
# model.add_seasonality(name='custom', period=365.25, fourier_order=18)  # 自定义季节性

model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True)
model.add_seasonality(name='custom', period=365.25, fourier_order=12)  # 自定义季节性

# 拟合模型
model.fit(ts)

# 创建未来时间序列
future = model.make_future_dataframe(periods=forecast_steps)

# 进行预测
forecast = model.predict(future)
print(forecast)
# 提取预测结果和置信区间
forecast_mean = forecast[['ds', 'yhat']].tail(forecast_steps).set_index('ds')['yhat']
confidence_interval = forecast[['ds', 'yhat_lower', 'yhat_upper']].tail(forecast_steps).set_index('ds')
# print(forecast_mean)
# 计算评估指标
time2 = pd.to_datetime(time1) + pd.DateOffset(days=1)  # time1 之后一天开始
time3 = time2 + pd.DateOffset(days=forecast_steps-1)  # 测试集为30天
time_test = pd.date_range(time2, time3)
ts_test = pd.DataFrame({'ds': pd.to_datetime(time_test), 'y': energy[col][time_test]}).set_index('ds')['y']
# print(ts_test)

mse = mean_squared_error(ts_test, forecast_mean)
mape = np.mean(np.abs((ts_test - forecast_mean) / ts_test)) * 100

# 打印评估指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 绘制原始数据、预测结果和置信区间
fig = model.plot(forecast)
predicted_index = forecast[['ds']].tail(forecast_steps).set_index('ds').index
# 原始数据中预测时间段的数据标红
plt.scatter(predicted_index, energy[col][predicted_index], color='red', label='Predicted Period')
plt.title('Prophet Forecast')
plt.show()
# 蓝色线 (Blue Line):
# 意义： 蓝色线代表模型对时间序列的预测值，即 "yhat" 列的值。这是模型对未来时间步的趋势的估计。
# 黑色点 (Black Dots):
## 意义： 黑色点表示历史观测值，即模型在拟合过程中使用的实际数据点。
# 蓝色背景块 (Blue Background Blocks):
# 意义： 蓝色背景块表示模型对未来时间步的不确定性，即预测的置信区间。
