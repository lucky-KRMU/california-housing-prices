'''
This is the python file which will be used to analyze the data.
'''
import pandas as pd

data = pd.read_csv('data/housing.csv')

print(type(data))
print(data.head())
print(data.info())