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
bedroom_mode = data['total_bedrooms'].mode()

print("Mean: ", bedroom_mean)
print("Median: ", bedroom_median)
print("Mode: ", bedroom_mode)
print("Maximum: ", bedroom_max)
print("Minimum: ", bedroom_min)

# Visualising the data through histogram and box plot
print(type(data['total_bedrooms']))
# plt.figure(figsize=(8,5))
# This is for directly using pandas
# data['total_bedrooms'].hist( bins=20, color='skyblue', edgecolor='black')
# for explicit matplotlib
plt.hist(data['total_bedrooms'], bins=100, color='skyblue', edgecolor='black', orientation='horizontal')
plt.title('Bedroom histogram')
plt.xlabel('Frequency')
plt.ylabel('Values')

plt.savefig('graphs/bedrooms_histogram.png')