# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
# plt.rcParams['axes.unicode_minus'] = False
#
# # 读取CSV文件
# df = pd.read_csv("C:/Users/ASUS/Downloads/比特币历史数据.csv")
#
# # 将日期列转换为数值型
# df['日期'] = pd.to_datetime(df['日期'])
# df['日期_numeric'] = pd.to_numeric(df['日期'])
#
# num = len(df)
#
# df['减半事件'] = 0
# for c, i in df.iterrows():
#     delta1 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2014-01-28')).days
#     delta2 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2017-09-09')).days
#     delta3 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2021-07-11')).days
#     delta4 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2025-06-17')).days
#     df.at[c, '减半事件'] = min(delta1, delta2, delta3, delta4)
#
# decay_parameter = 0.001
# weights = np.exp(-decay_parameter * np.array(df['减半事件']))
# normalized_weights = weights / np.sum(weights)
#
# # 创建一个DataFrame，以便更好地展示数据
# df['weights'] = normalized_weights * 100
# # 去除百分比符号 '%' 并将 '涨跌幅' 列转换为浮点数
# df['涨跌幅'] = df['涨跌幅'].str.rstrip('%').astype('float') / 100.0
#
# df['开盘'] = df['开盘'].str.replace(',', '').astype('float')
# df['收盘'] = df['收盘'].str.replace(',', '').astype('float')
# df['高'] = df['高'].str.replace(',', '').astype('float')
# df['低'] = df['低'].str.replace(',', '').astype('float')
#
# # 添加比特币减半事件的连续变量
# df['减半事件'] = (abs(pd.to_datetime(df['日期']) - pd.to_datetime('2012-11-28')) + abs(pd.to_datetime(df['日期']) -
#                   pd.to_datetime('2016-07-09')) + abs(pd.to_datetime(df['日期']) - pd.to_datetime('2020-05-11'))).dt.days
# df['减半事件'] /= df['减半事件'].max()  # 缩放到 [0, 1]
#
# columns_to_save = ['日期', '开盘', '收盘', '减半事件', 'weights']
# df[columns_to_save].to_csv('processed_bitcoin_data.csv', index=False)
#
# # 按照日期对数据集进行拆分
# split_date = pd.to_datetime('2022-02-01')  # 选择拆分日期
# train_data = df[df['日期'] < split_date]
# test_data1 = df[df['日期'] >= split_date]
# split_date1 = pd.to_datetime('2024-3-02')  # 选择拆分日期
# test_data = test_data1[test_data1['日期'] < split_date1]
#
# # 定义自变量和因变量
# X_train, X_test = train_data[['开盘', 'weights']], test_data[['开盘', 'weights']]
# Y_train, Y_test = train_data['收盘'], test_data['收盘']
#
# model = RandomForestRegressor(n_estimators=80, random_state=42)
# # model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 120, random_state = 42)
# model.fit(X_train, Y_train)
#
# # 预测结果
# train_predictions = model.predict(X_train)
# test_predictions = []
#
# # 使用前一个预测结果进行逐一预测
# last_prediction = X_test.iloc[0]['开盘']
# for i in range(20):
#     prediction = model.predict([[last_prediction, X_test.iloc[i]['weights']]])
#     print(prediction[0])
#     test_predictions.append(prediction[0])
#     last_prediction = prediction[0]
#
# # 计算均方误差
# mse = mean_squared_error(Y_test[:20], test_predictions)
# print('MSE:', mse)
#
# # 计算均方根误差
# rmse = np.sqrt(mse)
# print('RMSE:', rmse)
#
# # 计算平均绝对误差
# mae = mean_absolute_error(Y_test[:20], test_predictions)
# print('MAE:', mae)
#
# # 绘制实际涨跌幅和预测结果的对比图
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 21), Y_test[:20], label='True closing price')
# plt.plot(range(1, 21), test_predictions, label='Predicted closing price')
# # plt.title('Actual vs Predicted Returns')
# plt.xlabel('Days')
# plt.ylabel('Closing price')
# plt.legend()
# plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("C:/Users/ASUS/Downloads/比特币历史数据.csv")

df['收盘'] = df['收盘'].str.replace(',', '').astype('float')

# 将 'Date' 列转换为日期格式
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values('日期')

split_date = pd.to_datetime('2024-02-01')  # 选择拆分日期
df1 = df[df['日期'] < split_date]
# 去除百分比符号 '%' 并将 '涨跌幅' 列转换为浮点数
df1['涨跌幅'] = df1['涨跌幅'].str.rstrip('%').astype('float') / 100.0

df1['开盘'] = df1['开盘'].str.replace(',', '').astype('float')
df1['差'] = df1['收盘'] - df1['开盘']
# 使用 '涨跌幅' 列作为预测目标
data = df1[['收盘']].values.reshape(-1, 1)
# 数据归一化
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# 创建时间序列数据集的函数
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(np.hstack((data[i:(i + time_steps), 0], normalized_weights[i:(i + time_steps), 0])))
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# 选择时间步长
time_steps = 730

df1['减半事件'] = 0
for c, i in df.iterrows():
    delta1 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2014-01-28')).days
    delta2 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2017-09-09')).days
    delta3 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2021-07-11')).days
    delta4 = abs(pd.to_datetime(i['日期']) - pd.to_datetime('2025-06-17')).days
    df1.at[c, '减半事件'] = min(delta1, delta2, delta3, delta4)


decay_parameter = 0.001
weights = np.exp(-decay_parameter * np.array(df1['减半事件']))
normalized_weights = weights / np.sum(weights) * 100
# 创建一个DataFrame，以便更好地展示数据
# df['weights'] = normalized_weights * 100
normalized_weights = normalized_weights.reshape(-1, 1)

# 创建时间序列数据集
X, y = create_dataset(data_normalized, time_steps)

# 为LSTM输入调整数据的形状 (样本数，时间步长，特征数)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 建立LSTM模型
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(time_steps, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X, y, epochs=5, batch_size=32)

# 预测未来 'n' 个值
n = 30
future_dates = pd.date_range(start=df1['日期'].iloc[-1], periods=n+1, freq='D')[1:]
last_sequence = data_normalized[-time_steps:]
last_weights = weights[-time_steps:]
predicted_values = []

for _ in range(n):
    input_sequence = np.hstack((last_sequence.reshape(1, time_steps, 1), last_weights.reshape(1, time_steps, 1)))
    predicted_value = model.predict(input_sequence)[0, 0]
    predicted_values.append(predicted_value)
    last_sequence = np.append(last_sequence[1:], [[predicted_value]], axis=0)
    last_weights = np.append(last_weights[1:], [[np.exp(-0.005 * (future_dates[_] - future_dates[_-1]))]], axis=0)

# 将预测的值逆归一化为原始比例
predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
print(predicted_values)

# 计算MAE和RMSE
df2 = df[df['日期'] >= pd.to_datetime('2024-02-01')]
df2 = df2[df2['日期'] < pd.to_datetime('2024-03-02')]
true_values = df2['收盘'].values[-n:]
mae = mean_absolute_error(true_values, predicted_values)
rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
print("MAE:", mae)
print("RMSE:", rmse)

plt.figure(figsize=(12, 6))
plt.plot(df2['日期'], true_values, label='True closing price', marker='o')
plt.plot(future_dates, predicted_values, label='Predicted closing price', linestyle='dashed', marker='o')
plt.title('Bitcoin Predicted Price vs Actual Price')
plt.xlabel('Date')
plt.ylabel('Closing price')
# 设置 x 轴刻度
x_ticks = pd.date_range(start=df2['日期'].iloc[0], end=df2['日期'].iloc[-1], freq='7D')  # 每隔7天显示一个刻度
plt.xticks(x_ticks, rotation=45)
plt.legend()
plt.tight_layout()  # 自动调整子图参数，以便子图适合图形区域
plt.show()

