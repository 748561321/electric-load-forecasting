import pandas as pd

file_debug = 'C:/Users/city/Desktop/Graduation project/data/数据统合.csv'
db = pd.read_csv(file_debug,header= None,encoding='utf-8',low_memory=False)


file_path = 'C:/Users/city/Desktop/Graduation project/data/行列转置.csv'
df = pd.read_csv(file_debug,header= None,encoding='utf-8',low_memory=False)
data = df.values
data = list(map(list,zip(*data)))
data = pd.DataFrame(data)
data.to_csv(file_path,header=0,encoding="utf_8_sig",index=0)
# https://blog.csdn.net/chenpe32cp/article/details/82150074 pandas to_csv输出到文件的中文是乱码的问题
print(db)
print(data)
