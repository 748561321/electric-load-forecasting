import pandas as pd

file_debug = 'C:/Users/city/Desktop/Graduation project/data/area2-load.csv'
file_path = 'C:/Users/city/Desktop/Graduation project/data/weidu.csv'
df = pd.read_csv(file_debug,header= None,encoding='utf-8',low_memory=False)
print(df)
print(df.iloc[1096:2192,:-1])
time_data = df.iloc[1096:2192, :-1].values.tolist()

data = pd.DataFrame(time_data)
data.to_csv(file_path,header=0,encoding="utf_8_sig",index=0)
print(data)
# 之后补上第一行 和 几列内容
