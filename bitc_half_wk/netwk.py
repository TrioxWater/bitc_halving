"""
T
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv("C:/Users/ASUS/Downloads/比特币历史数据.csv")

# 将日期列转换为数值型
df['日期'] = pd.to_datetime(df['日期'])
df['日期_numeric'] = pd.to_numeric(df['日期'])

# 去除百分比符号 '%' 并将 '涨跌幅' 列转换为浮点数
df['涨跌幅'] = df['涨跌幅'].str.rstrip('%').astype('float') / 100.0

df['开盘'] = df['开盘'].str.replace(',', '').astype('float')
df['收盘'] = df['收盘'].str.replace(',', '').astype('float')
df['高'] = df['高'].str.replace(',', '').astype('float')
df['低'] = df['低'].str.replace(',', '').astype('float')

# 添加比特币减半事件的连续变量
df['减半事件'] = (abs(pd.to_datetime(df['日期']) - pd.to_datetime('2012-11-28')) + abs(pd.to_datetime(df['日期']) -
                  pd.to_datetime('2016-07-09')) + abs(pd.to_datetime(df['日期']) - pd.to_datetime('2020-05-11'))).dt.days
df['减半事件'] /= df['减半事件'].max()  # 缩放到 [0, 1]

# 选择自变量和因变量
X = df[['减半事件', '开盘']]
Y = df['收盘']

# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# 使用模型进行预测
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# # 绘制实际收盘价和预测收盘价的对比图（训练集）
# plt.figure(figsize=(12, 6))
# plt.plot(df['日期'].iloc[:len(Y_train)], Y_train, label='实际收盘价 (训练集)')
# plt.plot(df['日期'].iloc[:len(Y_train)], train_predictions, label='预测收盘价 (训练集)')
# plt.title('实际收盘价 vs 预测收盘价 (训练集)')
# plt.xlabel('日期')
# plt.ylabel('收盘价')
# plt.legend()
# plt.show()

# 绘制实际收盘价和预测收盘价的对比图（测试集）
plt.figure(figsize=(12, 6))
plt.plot(df['日期'].iloc[-len(Y_test):], Y_test, label='实际收盘价 (测试集)')
# plt.plot(df['日期'].iloc[-len(Y_test):], test_predictions, label='预测收盘价 (测试集)')
plt.title('实际收盘价 vs 预测收盘价 (测试集)')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.legend()
plt.show()
