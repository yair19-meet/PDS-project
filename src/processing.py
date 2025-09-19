import pandas as pd
import numpy as np

data = pd.read_csv('city_data.csv', delimiter='|')
new_header = data.iloc[0]
data = data[1:]
data.columns = new_header
cities = []
states = []
for i in range(len(data)):
    if ',' in data.iloc[i, :]["City"]:
        city_and_state = data.iloc[i, :]["City"].split(",")
    elif '.' in data.iloc[i, :]["City"]:
         city_and_state = data.iloc[i, :]["City"].split(".")
    else:
        city_and_state = data.iloc[i, :]["City"].split(";")
         
    cities.append(city_and_state[0])
    if (len(city_and_state)) > 1:
        states.append(city_and_state[1])
    else:
        states.append("")

# data.reset_index(drop=True)

data.drop(columns=['City'], inplace=True)

data.insert(0, 'City', cities)

data.insert(1, 'Country', states)

data.drop(columns=['Average Price Groceries'], inplace=True)

data.fillna('0', inplace=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print(data.head(83))


