import pandas as pd
#  截取五年的数据 删掉多余的内容W
file_debug = 'C:/Users/city/Desktop/Graduation project/data/area2-load.csv'
file_path = 'C:/Users/city/Desktop/Graduation project/data/five_years.csv'
df = pd.read_csv(file_debug,header= None,encoding='utf-8',low_memory=False)
print(df)
print(df.iloc[:2192,:-7])
time_data = df.iloc[:2192, :-7].values.tolist()

data = pd.DataFrame(time_data)
data.to_csv(file_path,header=0,encoding="utf_8_sig",index=0)
print(data)
