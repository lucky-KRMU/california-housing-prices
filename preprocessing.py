'''
This is the python file which will be used to analyze the data.
'''
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/housing.csv')

print(type(data))
print(data.head())
print(data.info())

# investigating bedrooms
bedroom_mean = data['total_bedrooms'].mean()
bedroom_median = data['total_bedrooms'].median()
bedroom_min = data['total_bedrooms'].min()
bedroom_max = data['total_bedrooms'].max()

print("Mean: ", bedroom_mean)
print("Median: ", bedroom_median)
print("Maximum: ", bedroom_max)
print("Minimum: ", bedroom_min)

plt.plot(data['total_bedrooms'])
plt.show()