import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
# 搜索sarima最优参数
energy = pd.read_csv('C:/Users/city/Desktop/Graduation project/data/数据统合单变量.csv')
energy['YMD'] = pd.to_datetime(energy['YMD'], format='%Y%m%d')
energy.set_index('YMD', inplace=True)
time_index = pd.date_range('2012-01-01', '2014-12-31', freq='D')
ts = pd.Series(energy['0:00:00'], index=time_index)

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
# 当前 ARIMA(0, 0, 0)x(0, 2, 2, 90) - AIC:13472.800454036511
# ARIMA(0, 0, 1)x(0, 1, 2, 90) - AIC:13822.969862696667
# ARIMA(0, 0, 1)x(0, 2, 2, 90) - AIC:12709.286147524763

# 跑到 ARIMA(0, 0, 0)x(2, 1, 2, 90) - AIC:14074.393221233291 停了

pdq_x_PDQs = [(x[0], x[1], x[2], 90) for x in list(itertools.product(p, d, q))]
a=[]
b=[]
c=[]
wf=pd.DataFrame()
for param in pdq:
    seasonal_param = (0,2,2,90)
    # for seasonal_param in pdq_x_PDQs:
    #     if  param == (0,0,0):
    #         print('skip')
    #         continue
    try:
        mod = sm.tsa.statespace.SARIMAX(ts,order=param,seasonal_order=seasonal_param,enforce_stationarity=False,enforce_invertibility=False)
        results = mod.fit()
        print('ARIMA{}x{} - AIC:{}'.format(param, seasonal_param, results.aic))
        a.append(param)
        b.append(seasonal_param)
        c.append(results.aic)
    except:
        continue
wf['pdq']=a
wf['pdq_x_PDQs']=b
wf['aic']=c
print(wf[wf['aic']==wf['aic'].min()])

# 网格搜索结果：            pdq     pdq_x_PDQs           aic
#                    386  (1, 1, 2)  (0, 2, 2, 12)  16010.136494

# p = d = q = range(0, 3)
# pdq = list(itertools.product(p, d, q))
#
# best_rmse = float('inf')
# best_order = None
#
# for order in pdq:
#     try:
#         model = ARIMA(ts, order=order)
#         results = model.fit()
#         rmse = np.sqrt(((results.fittedvalues - ts) ** 2).mean())
#
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_order = order
#     except:
#         continue
#
# print(f"Best ARIMA Order: {best_order} with RMSE: {best_rmse}")