import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# 绘制图形
plt.figure(figsize=(12, 6))

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

    # # 使用KMeans将数据分为两类
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # kmeans.fit(closing_prices)
    # labels = kmeans.labels_
    #
    # # 将聚类结果添加到数据框中
    # df_subset['Cluster'] = labels
    #
    # # 根据聚类结果绘制收盘价格的分布图
    # plt.figure(figsize=(12, 6))
    # for cluster in range(2):
    #     cluster_data = df_subset[df_subset['Cluster'] == cluster]
    #     plt.scatter(cluster_data.index, cluster_data['收盘'], label=f'Cluster {cluster}')
    #
    # # 设置图形属性
    # plt.xlabel('Index')
    # plt.ylabel('Closing Price')
    # plt.title('Bitcoin Closing Price Distribution by Clusters')
    # plt.legend()
    # plt.show()

    # 计算肘部法则下的最佳簇数量
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(closing_prices)
        sse.append(kmeans.inertia_)

    # 绘制SSE随簇数量变化的曲线
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal K')
    plt.show()

    # # 绘制收盘价格和时间的关系
    # plt.plot(range(len(df_subset)), df_subset['收盘'], label=f'{start_date} to {end_date}')
    #
    # plt.xlabel('Time')
    # plt.ylabel('Closing Price')
    # plt.title('Bitcoin Closing Price Over Time')
    # # 设置纵坐标刻度
    # # min_price = df_subset['收盘'].min()
    # # max_price = df_subset['收盘'].max()
    # # plt.yticks(range(int(min_price / 1000) * 1000, int(max_price / 1000) * 1000 + 1000, 1000))
    # plt.legend()
    # plt.show()