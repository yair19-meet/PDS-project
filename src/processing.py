import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#reading file
csv_path = Path(__file__).parent.parent / 'city_data.csv'
data = pd.read_csv(csv_path, delimiter='|')

#replacing the columns with first row
new_header = data.iloc[0]
data = data[1:]
data.columns = new_header


#spliting the City column to two seperate columns of City and Country
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

data.drop(columns=['City'], inplace=True)
data.insert(0, 'City', cities)
data.insert(1, 'Country', states)


#removing a column with missing values
data.drop(columns=['Average Price Groceries'], inplace=True)

#replacing missing values with zeros
data.fillna('0', inplace=True)

#removing duplicate rows
data.drop_duplicates(inplace=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

#changing type of columns from string to int
data['Population'] = data['Population'].astype(np.int64)
data['Average Monthly Salary'] = data['Average Monthly Salary'].astype(np.float64)
data['Average Cost of Living'] = data['Average Cost of Living'].astype(np.float64)
data['Avgerage Rent Price'] = data['Avgerage Rent Price'].astype(np.float64)
data['GDP per Capita'] = data['GDP per Capita'].astype(np.float64)
data['Unemployment Rate'] = data['Unemployment Rate'].astype(np.float64)
data['Youth Dependency Ratio'] = data['Youth Dependency Ratio'].astype(np.float64)
data['Working Age Population '] = data['Working Age Population '].astype(np.float64)
data['Population Density'] = data['Population Density'].astype(np.float64)

#displaying the total population of each country in the data set
population_ser = data.groupby('Country')['Population'].sum().sort_values(ascending=False)
population_summary = pd.DataFrame({"Population Total": population_ser})
print("\nPopulation Summary: \n")
print(population_summary)

#displaying data about quantity of cities
cities_count_ser = data.groupby('Country')['City'].count().sort_values(ascending=False)
amount_of_cities = pd.DataFrame({"Number of Cities": cities_count_ser})
total_number_of_cities = amount_of_cities['Number of Cities'].sum()
print("\nCities Summery:\n")
print(amount_of_cities)
print(f"\nAmount of Cities in total: {total_number_of_cities}\n")


#displaying data about cities with high salaries
print("\nHigh Salary Cities:\n")
print(data[data["Average Monthly Salary"] > 1600].sort_values(by='Average Monthly Salary', ascending=False)[["City", "Average Monthly Salary"]])

#displaying data about cities with low cost of living
print("\nLow Cost of living:\n")
print(data[data["Average Cost of Living"] < 900].sort_values(by="Average Cost of Living", ascending=False)[["City", "Average Cost of Living"]])

#calculating difference between average salary and average cost of living in cities
data["avg salary - avg cost of living"] = data["Average Monthly Salary"] - data["Average Cost of Living"]
print("\nDifference between cost of living and salary:\n")
print(data[['City', "avg salary - avg cost of living"]].sort_values(by='avg salary - avg cost of living', ascending=False).head(5))

#plot for showing correlation between Salary and Cost of Living
# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# ax.scatter(data["Average Monthly Salary"], data["Average Cost of Living"])
# ax.set_xlabel("Salary")
# ax.set_ylabel("Cost of living")
# ax.set_title("Salary versus Cost of Living")
# plt.show()

plt.scatter(x=data["Average Monthly Salary"], y=data["Average Cost of Living"])
plt.xlabel('Salary')
plt.ylabel('Cost of Living')
plt.show()

languages = {}
for item in data['Main Spoken Languages']:
    lst = item.split(",")
    for i in lst:
        i = i.strip()
        if i not in languages:
            languages[i] = 1
        else:
            languages[i] += 1

languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)

spoken_languages, counts = map(list, zip(*languages))

# ax2.bar(x=spoken_languages, height=counts, width=0.5)
# ax2.set_title("Spoken languages")
# plt.xticks(rotation=45)
# plt.show()

plt.bar(x=spoken_languages, height=counts)
plt.xticks(rotation=45)
plt.xlabel('Language')
plt.ylabel('Amount of Cities')
plt.show()

# print(data[data['City'] == "Lisbon"][["Average Monthly Salary", "Average Cost of Living"]])
# print(data[data['City'] == "Zurich"][["Average Monthly Salary", "Average Cost of Living"]])
# print(data[data['City'] == "Barcelona"][["Average Monthly Salary", "Average Cost of Living"]])