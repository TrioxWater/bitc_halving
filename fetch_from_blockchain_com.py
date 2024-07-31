"""
This script is used to fetch data from blockchain.com.
By changing the URL and params, you can fetch different data.
For detailed information, please refer to the official documentation:
"""
import requests
import os
URL = "https://api.blockchain.info/charts/n-payments"

params = {
    "timespan": "all",
    "sampled": "false",
    "metadata": "false",
    "daysAverageString": "1d",
    "cors": "true",
    "format": "json"
}

response = requests.get(URL, params=params)
print(response.status_code)

# write the response to a file
with open("responce.json", "w") as f:
    f.write(response.text)
    f.close()

# run deal.py

