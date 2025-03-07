import pandas_datareader.data as pdr
from datetime import datetime

"""
### Get bitcoin data
"""
end = datetime.today()
start = datetime(end.year-4,end.month,end.day)
df = pdr.DataReader('BTC-USD','yahoo',start, end)

df.head()