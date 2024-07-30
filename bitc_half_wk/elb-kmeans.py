from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

# 读取比特币历史数据
df = pd.read_csv("C:/Users/ASUS/Downloads/比特币历史数据.csv")

# 将 '日期' 列转换为日期格式
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values('日期')

df['收盘'] = df['收盘'].str.replace(',', '').astype('float')

split_date = pd.to_datetime('2012-11-28')  # 选择拆分日期
df = df[df['日期'] >= split_date]

# 划分时间段
split_dates = ['2016-07-09', '2020-05-11', df['日期'].iloc[-1]]

# 遍历时间段
for i in range(len(split_dates)):
    if i == 0:
        start_date = df['日期'].iloc[0]
    else:
        start_date = pd.to_datetime(split_dates[i-1]) + pd.Timedelta(days=1)
    end_date = pd.to_datetime(split_dates[i])

    # 获取时间段内的数据
    df_subset = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]

    # 提取收盘价格
    closing_prices = df_subset['收盘'].values.reshape(-1, 1)

    # 对收盘价格进行标准化处理
    closing_prices_scaled = TimeSeriesScalerMeanVariance().fit_transform(closing_prices)
    #
    # # 将三维数组转换为二维数组
    # closing_prices_flat = closing_prices_scaled.reshape(closing_prices_scaled.shape[0], closing_prices_scaled.shape[1])
    #
    # # 计算不同聚类数目的轮廓系数
    # silhouette_scores = []
    # for k in range(2, 12):
    #     km = TimeSeriesKMeans(n_clusters=k, verbose=True, random_state=42)
    #     km.fit(closing_prices_scaled)
    #     labels = km.labels_
    #     # 检查聚类结果是否只有一个类别
    #     unique_labels = len(np.unique(labels))
    #     if unique_labels == 1:
    #         print(f"Skipping silhouette score calculation for {k} clusters because only one cluster found.")
    #         continue
    #
    #     silhouette_scores.append(silhouette_score(closing_prices_scaled, labels))
    #
    # # 检查是否存在有效的轮廓系数，如果没有则给出提示
    # if len(silhouette_scores) == 0:
    #     print("No valid silhouette scores calculated. Adjust clustering parameters or data preprocessing.")
    # else:
    #     # 绘制轮廓系数随簇数量变化的曲线
    #     plt.plot(range(2, 11), silhouette_scores, marker='o')
    #     plt.xlabel('Number of Clusters')
    #     plt.ylabel('Silhouette Score')
    #     plt.title('Silhouette Score for Optimal K')
    #     plt.show()

    # 使用K-Means Time Series Clustering进行聚类
    n_clusters = 3  # 假设我们要分为2类
    km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, random_state=42)
    km.fit(closing_prices_scaled)

    # 获取每个时间序列的聚类标签
    labels = km.labels_

    # 将聚类标签添加到DataFrame中
    df_subset['Cluster'] = labels

    # 绘制聚类结果
    plt.figure(figsize=(12, 6))
    for i in range(n_clusters):
        cluster_i = df_subset[df_subset['Cluster'] == i]
        plt.scatter(cluster_i['日期'], cluster_i['收盘'], label=f'Cluster {i}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Bitcoin Clustering by Time Series')
    plt.legend()
    plt.show()
