# 币价文件:csv, Date,Open,High,Low,Close,Adj Close,Volume
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os

if os.path.exists('DigitCur-USD'):
    os.chdir('DigitCur-USD')

# 读取csv文件
df = pd.read_csv('BTC-USD.csv')
# 币价取开盘价（网站的其他参数都给的是早八点的）
price = df['Open']
# 日期，%Y/%m/%d
date = df['Date']
date = [time.mktime(time.strptime(i, '%Y/%m/%d')) for i in date]

maxPrice = max(price)
minPrice = min(price)
price = [(i - minPrice) / (maxPrice - minPrice) for i in price] # 归一化

legends = []

def addLine(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        # x是时间, Y-M-D H:M:S y是待参考的数据
        data_pair = data['values']
        name = data['name'] + '/' + data['unit']
        for i in data_pair:
            i['x'] = time.mktime(time.strptime(i['x'], '%Y-%m-%d %H:%M:%S'))
        # 踢掉早于币价的数据
        data_pair = [i for i in data_pair if i['x'] >= date[0] and i['x'] <= date[-1]]
        data = [i['y'] for i in data_pair]
        maxData = max(data)
        minData = min(data)
        data = [(i - minData) / (maxData - minData) for i in data]
        time_a = [i['x'] for i in data_pair]

    plt.plot(time_a, data)
    legends.append(name)

def addPoint(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        # x是时间, Y-M-D H:M:S y是待参考的数据
        data_pair = data['values']
        name = data['name'] + '/' + data['unit']
        for i in data_pair:
            i['x'] = time.mktime(time.strptime(i['x'], '%Y-%m-%d %H:%M:%S'))
        # 踢掉早于币价的数据
        data_pair = [i for i in data_pair if i['x'] >= date[0] and i['x'] <= date[-1]]
        data = [i['y'] for i in data_pair]
        maxData = max(data)
        minData = min(data)
        data = [(i - minData) / (maxData - minData) for i in data]
        time_a = [i['x'] for i in data_pair]

    plt.plot(time_a, data, 'ro', markersize=0.05)
    legends.append(name)

# with open('miners-revenue.json', 'r') as f:
#     data = json.load(f)
#     # x是时间, Y-M-D H:M:S y是待参考的数据
#     miners_revenue_pair = data['values']
#     for i in miners_revenue_pair:
#         i['x'] = time.mktime(time.strptime(i['x'], '%Y-%m-%d %H:%M:%S'))
#     unit = data['unit']
#     # 踢掉早于币价的数据
#     miners_revenue_pair = [i for i in miners_revenue_pair if i['x'] >= date[0] and i['x'] <= date[-1]]
#     miners_revenue = [i['y'] for i in miners_revenue_pair]
#     maxRevenue = max(miners_revenue)
#     minRevenue = min(miners_revenue)

#     miners_time = [i['x'] for i in miners_revenue_pair]
#     miners_revenue = [(i - minRevenue) / (maxRevenue - minRevenue) for i in miners_revenue] # 归一化

addLine('miners-revenue.json')
addLine('hash-rate.json')

# addPoint('transactions-per-sec.json')

bitcoin_halving_indicators_x = ['2012-11-28', '2016-7-9', '2020-5-11']
bitcoin_halving_indicators_y = [0.5, 0.25, 0.125]
bitcoin_halving_indicators_x = [time.mktime(time.strptime(i, '%Y-%m-%d')) for i in bitcoin_halving_indicators_x]


# 画到同一张图上
plt.plot(date, price, bitcoin_halving_indicators_x, bitcoin_halving_indicators_y, 'ro')
plt.xlabel('date, shown in timestamp')
plt.ylabel('normalized value')
plt.title('bitcoin analysis')
plt.legend(legends + ['BTC-USD', 'bitcoin halving indicators'])
plt.show()

# save
# plt.savefig('bitcoin-minersRevenue.png')









    

