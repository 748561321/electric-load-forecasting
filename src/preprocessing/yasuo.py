import pandas as pd
# 压缩到一列 取1天24分 即除4
file_debug = 'C:/Users/city/Desktop/Graduation project/data/five_years.csv'
file_path = 'C:/Users/city/Desktop/Graduation project/data/one_yearsp2.csv'
df = pd.read_csv(file_debug,header= None,encoding='utf-8',low_memory=False)
# print(df)
# print(df.iloc[1:, 1:])
# 1:-6 : 20120101 - 20150104
data = df.iloc[1:, 1:].values.tolist()

rows, columns = df.shape
time = []
cnt =-1
for i in range(rows-365,rows):
    for j in range(1,columns):
        cnt += 1
        if cnt % 4 == 0:
            str =  df[0][i]  + " " + df[j][0]
            time.append(str)

merged_data = []
cnt =-1
for i in range(len(data)):
    for j in range(len(data[i])):
        cnt += 1
        if cnt % 4 == 0:
            merged_data.append(data[i][j])

time_data = list(zip(time, merged_data))
data = pd.DataFrame(time_data)
data.to_csv(file_path,header=0,encoding="utf_8_sig",index=0)

# print(data)
