import pandas as pd
import numpy as np

data = pd.read_csv('city_data.csv', delimiter='|')
new_header = data.iloc[0]
data = data[1:]
data.columns = new_header
print(data)


