import pandas as pd
file_debug = 'C:/Users/city/Desktop/Graduation project/data/weidu.csv'
file_path = 'C:/Users/city/Desktop/Graduation project/data/weidusp2.csv'
df = pd.read_csv(file_debug,header= None,encoding='utf-8',low_memory=False)

from datetime import datetime
import holidays

def isholiday(date_obj):
    cn_holidays = holidays.China()
    if date_obj.weekday() >= 5 or date_obj in cn_holidays:
        return 1
    return 0

def parse_date_number(date_number):
    # 解析日期数字
    date_number = int(date_number)
    year = int(date_number // 10000)
    month = int((date_number // 100) % 100)
    day = int(date_number % 100)

    # 创建日期对象
    date_obj = datetime(year, month, day)

    # 获取所需信息
    hourofday = date_obj.hour
    dayofweek = date_obj.weekday()  # Monday is 0 and Sunday is 6
    dayofmonth = date_obj.day
    weekofyear = date_obj.isocalendar()[1]  # ISO week number
    month = date_obj.month
    season = (month % 12 + 3) // 3  # Calculate season (1: Winter, 2: Spring, 3: Summer, 4: Autumn)
    year = date_obj.year
    holiday= isholiday(date_obj)

    return hourofday, dayofweek, dayofmonth, weekofyear, month, season, year,holiday

rows, columns = df.shape
# print(rows)   1097
# print(columns)    107
# print(df[columns-10][0])
time = []
cnt =-1
hourofday, dayofweek, dayofmonth, weekofyear, month, season, year, holiday = [], [], [], [], [], [], [], []
for i in range(3,rows):
    for j in range(1,columns-10):
        cnt += 1
        if cnt % 4 == 0:

            str =  df.iloc[i,0]  + " " + df.iloc[0,j]
            time.append(str)

            parsed_values = parse_date_number(df.iloc[i, 0])
            hourofday.append(parsed_values[0])
            dayofweek.append(parsed_values[1])
            dayofmonth.append(parsed_values[2])
            weekofyear.append(parsed_values[3])
            month.append(parsed_values[4])
            season.append(parsed_values[5])
            year.append(parsed_values[6])
            holiday.append(parsed_values[7])


# #
merged_data = []
cnt =-1
for i in range(3,rows):
    for j in range(1,columns-10):
        cnt += 1
        if cnt % 4 == 0:
            merged_data.append(df.iloc[i,j])

ycharge = []
yaverage,ypeak,ytrough,yrange = [],[],[],[]
hT,lT,aT,rT = [],[],[],[]
humudityrange,precipitation = [],[]
for i in range(3,rows):
    for j in range(1,24+1):
        ycharge.append(df.iloc[i-2,j])
        yaverage.append(df.iloc[i-1,columns-10])
        ypeak.append(df.iloc[i-1,columns-9])
        ytrough.append(df.iloc[i-1,columns-8])
        yrange.append(df.iloc[i-1,columns-7])
        hT.append(df.iloc[i-1,columns-6])
        lT.append(df.iloc[i-1,columns-5])
        aT.append(df.iloc[i-1,columns-4])
        rT.append(df.iloc[i-1,columns-3])
        humudityrange.append(df.iloc[i-1,columns-2])
        precipitation.append(df.iloc[i-1,columns-1])
#  平均电荷 最高电荷 最低电荷 电荷极差 最高温度 最低温度 平均温度 温度极差



time_data = list(zip(
    time, merged_data,
    ycharge,
yaverage,ypeak,ytrough,yrange,
hT,lT,aT,rT,
humudityrange,precipitation,
hourofday, dayofweek, dayofmonth, weekofyear, month, season, year,holiday))

data = pd.DataFrame(time_data)

column = [
    "date",
    "charge",

    "48h_back_charge",
    "avg_charge",
    "peak_charge",
    "trough_charge",
    "range_charge",

    "highest_temp",
    "lowest_temp",
    "avg_temp",
    "range_temp",

    "humidity",
    "precipitation",

    "hourofday",
    "dayofweek",
    "dayofmonth",
    "weekofyear",

    "month",
    "quarter",
    "year",
    "holiday"
]
data.columns = column

data.to_csv(file_path,header=1,encoding="utf_8_sig",index=0)
print(data)
# 1.当前电荷
# 2.48小时前电荷
# 3.昨日平均电荷
# 4.昨日电荷峰值
# 5.昨日电荷低谷
# 6.昨日电荷极差
# 7.最高温度℃
# 8.最低温度℃
# 9.平均温度℃
# 10.日温度极差
# 11.日相对湿度
# 12.日降雨量
# 13.一天中的第几个小时
# 14.一周中的第几天
# 15.一月中的第几天
# 16.一年中的第几周
# 17.月份
# 18.季度
# 19.年份
# 20.是否为节假日（包括周末

# data = pd.DataFrame(time_data)
# data.to_csv(file_path,header=0,encoding="utf_8_sig",index=0)
# print(data)