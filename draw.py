import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('DigitCur-USD/BTC-USD.csv')

# Extract the required attributes
date = data['Date']
open_price = data['Open']
high = data['High']
low = data['Low']
close = data['Close']
adj_close = data['Adj Close']
volume = data['Volume']

# Plot the data
plt.plot(date, high, label='High')
plt.plot(date, low, label='Low')
plt.plot(date, close, label='Close')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('BTC-USD Data')
plt.legend()

# Show the plot
plt.show()

plt.figure()
plt.plot(date, (close - open_price) / open_price, label='Change')
plt.xlabel('Date')
plt.ylabel('Change')
plt.title('BTC-USD Data')
plt.legend()
plt.show()


