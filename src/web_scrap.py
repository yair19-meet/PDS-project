# Import scrapy
import scrapy

# Import the CrawlerProcess: for running the spider
from scrapy.crawler import CrawlerProcess

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




lst = []

countries = list(data['Country'])
cities = list(data['City'])


# Create the Spider class
class geo_Spider(scrapy.Spider):
  name = "geo_spider"
  coordinates = {}
  # start_requests method
  def start_requests(self):
    yield scrapy.Request(url = 'https://en.wikipedia.org/wiki/Main_Page',
                         callback = self.parse_1)
  # First parsing method
  def parse_1(self, response):
    url = response.xpath('//li[@id="n-contents"]/a/@href').get()
    yield response.follow(url = url,
                            callback = self.parse_2)
  # Second parsing method
  def parse_2(self, response):
    url = response.xpath('//a[contains(@title, "Geography")]/@href').get()
    yield response.follow(url = url, callback = self.parse_3)

  def parse_3(self, response):
    for country in countries:
      url = response.xpath(f'//a[contains(@title, "{country}")]/@href').get()
      if url:
          yield response.follow(url, callback=self.parse_4)
      else:
          self.logger.warning(f"No page found for country: {country}")
      

  def parse_4(self, response):
      url = response.xpath('//a[contains(@title, "List of cities")]/@href').get()
      if url:
        yield response.follow(url, callback=self.parse_5)
      else:
        self.logger.warning(f"No city list found on {response.url}")

  def parse_5(self, response):
    for city in cities:
      url = response.xpath(f'//a[contains(@title, "{city}")]/@href').get()
      if url:
        yield response.follow(url, callback=self.parse_6)
      else:
        self.logger.warning(f"No page found for city: {city}")

  def parse_6(self, response):
    coord = {"latitude" : response.xpath('//span[contains(@class, "latitude")]/text()').extract_first(), \
      "longitude" : response.xpath('//span[contains(@class, "longitude")]/text()').extract_first()}
    city = response.xpath('//span[contains(@class, "mw-page-title-main")]/text()').extract_first()
    self.coordinates[city] = coord



# Run the Spider
process = CrawlerProcess()
process.crawl(geo_Spider)
process.start()

print("Coordinates: \n\n\n")
#print(geo_Spider.coordinates)

df = pd.DataFrame.from_dict(geo_Spider.coordinates, orient='index')
print(df)

