import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

file_path = 'C:/Users/city/Desktop/Graduation project/data/five_yearsp2.csv'
df = pd.read_csv(file_path, header=None, encoding='utf-8', low_memory=False)

data = df.iloc[:, 1].values


def detect_outliers_and_correct(start_index):
    subset_df = df.iloc[start_index:start_index + 90 * 24]

    mean = subset_df.mean(axis=0)
    std = subset_df.std(axis=0)

    # 计算子集的上下限
    upper_limit = mean + 3 * std
    lower_limit = mean - 3 * std

    # 使用 align() 方法手动对齐列，以避免自动重新索引的警告
    subset_df, lower_limit = subset_df.align(lower_limit, axis=1, join='inner')
    subset_df, upper_limit = subset_df.align(upper_limit, axis=1, join='inner')

    # 找到异常值的索引
    outlier_indices = ((subset_df < lower_limit) | (subset_df > upper_limit))

    # 打印异常值
    outliers = subset_df[outlier_indices.any(axis=1)]
    if not outliers.empty:
        print(f"在行 {start_index } 到 {start_index + 90*24} 中的异常值：")
        print(outliers)

        # 使用线性插值纠正异常值
        subset_df[outlier_indices] = np.nan  # 将异常值置为NaN
        subset_df = subset_df.interpolate(method='linear', axis=0)  # 使用线性插值填充NaN值

        # 重新打印纠正后的数据
        print(f"在行 {start_index} 到 {start_index + 90*24} 中的纠正后数据：")
        print(subset_df[outlier_indices.any(axis=1)])


for i in range(0, len(df), 90 * 24):
    detect_outliers_and_correct(i)
