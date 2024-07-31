import os
from datetime import datetime
import pandas as pd

def normalize(data: list):
    """
    Normalize the data to [0, 1]
    """
    min_data = min(data)
    max_data = max(data)
    return [(i - min_data) / (max_data - min_data) for i in data]

# 读取csv文件
df = pd.read_csv('BTC-USD.csv')


# 币价取开盘价
price = df['Open']
# 日期，%Y/%m/%d
date = df['Date']
date = [datetime.strptime(i, '%Y/%m/%d') for i in date]

# dPrice = [price[i] - price[i - 1] for i in range(1, len(price))]
bitcoin_price  = price

bitcoin_halving_date = ['2012-11-28', '2016-7-9', '2020-5-11']
bitcoin_halving_date = [datetime.strptime(i, '%Y-%m-%d') for i in bitcoin_halving_date]

bitcoin_price_derivative = normalize([bitcoin_price[i] - bitcoin_price[i - 1] for i in range(1, len(bitcoin_price))])

date = date[1:]

date = date[1::10]
bitcoin_price_derivative = bitcoin_price_derivative[1::10]


# 绘制derivative图

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

fig, ax = plt.subplots()
# 细线
ax.plot(date, bitcoin_price_derivative, label='Derivative', color='black', alpha=0.5)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Year')
ax.set_ylabel('Normalized Difference (10Days)')
ax.axvline(x=bitcoin_halving_date[1], color='g', linestyle='--', label='Second Halving')
ax.axvline(x=bitcoin_halving_date[2], color='b', linestyle='--', label='Third Halving')
plt.show()


