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
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# 选择时间步长
time_steps = 365
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
model.fit(X, y, epochs=20, batch_size=32)

# 预测未来 'n' 个值
n = 30
future_dates = pd.date_range(start=df1['日期'].iloc[-1], periods=n+1, freq='D')[1:]
last_sequence = data_normalized[-time_steps:]
predicted_values = []

for _ in range(n):
    input_sequence = last_sequence.reshape(1, time_steps, 1)
    predicted_value = model.predict(input_sequence)[0, 0]
    predicted_values.append(predicted_value)
    last_sequence = np.append(last_sequence[1:], [[predicted_value]], axis=0)

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

