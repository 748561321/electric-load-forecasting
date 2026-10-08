import pandas as pd
# 压缩到一列
file_debug = 'C:/Users/city/Desktop/Graduation project/data/five_years.csv'
file_path = 'C:/Users/city/Desktop/Graduation project/data/five_yearsp.csv'
df = pd.read_csv(file_debug,header= None,encoding='utf-8',low_memory=False)
print(df)
print(df.iloc[1:, 1:])
# 1:-6 : 20120101 - 20150104
data = df.iloc[1:, 1:].values.tolist()
rows, columns = df.shape
time = []
for i in range(1,rows):
    for j in range(1,columns):
        str =  df[0][i]  + " " + df[j][0]
        time.append(str)

merged_data = [item for sublist in data for item in sublist]

time_data = list(zip(time, merged_data))
data = pd.DataFrame(time_data)
data.to_csv(file_path,header=0,encoding="utf_8_sig",index=0)
print(data)
