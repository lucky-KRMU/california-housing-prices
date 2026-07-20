'''
This is the python file which will be used to analyze the data.
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

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

# splitting the data

# train_data, temp_data = train_test_split(data, 
#                                         test_size=.2,
#                                         random_state=42,
#                                         stratify=data['median_income'])

# validation_data, test_data = train_test_split(temp_data,
#                                               test_size=.5,
#                                               random_state=42,
#                                               stratify=data['median_income'])
# # Since this is a large dataset, so stratification isn't required but we are still doing it as it's a good pracatice.

# print(train_data.info())
# print(validation_data.info())
# print(test_data.info())

# we can't do stratification here because stratification is done for categorical classes classification and not of simple continuous values.


# splitting the data

train_data, temp_data = train_test_split(data, 
                                        test_size=.2,
                                        random_state=42)

validation_data, test_data = train_test_split(temp_data,
                                              test_size=.5,
                                              random_state=42)

print(train_data.info())
print(validation_data.info())
print(test_data.info())

# refining the data
x_train = train_data.drop('median_house_value', axis=1)
y_train = train_data['median_house_value']

x_validation = validation_data.drop('median_house_value', axis=1)
y_validation = validation_data['median_house_value']

x_test = test_data.drop('median_house_value', axis=1)
y_test = test_data['median_house_value']

print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)
print(x_test.shape)
print(y_test.shape)



# splitting the data appropriately for imputation and encoding
x_train_housing = x_train.drop('ocean_proximity', axis=1)
x_validation_housing = x_validation.drop('ocean_proximity', axis=1)
x_test_housing = x_test.drop('ocean_proximity', axis=1)

x_train_proximity = x_train[['ocean_proximity']] # We are using double brackets here just to make sure that 
# x_train_proximity is dataframe instead of a series. since series are unidimensional and dataframes are two dimensional, 
# as in unicoding it would expect two dimensional dataframe to perform that particular operation
x_validation_proximity = x_validation[['ocean_proximity']]
x_test_proximity = x_test[['ocean_proximity']]

# applying imputation
imputer = SimpleImputer(strategy='median')

# teaching the imputer
imputer.fit(x_train_housing)

# updating the values
x_train_housing_tr = imputer.transform(x_train_housing)
x_validation_housing_tr = imputer.transform(x_validation_housing)
x_test_housing_tr = imputer.transform(x_test_housing)

# since imputer.transform() returns a numpy array, we will now explicitly convert it into dataframe.
x_train_housing_tr = pd.DataFrame(
    x_train_housing_tr,
    columns=x_train_housing.columns,
    index=x_train_housing.index
)

x_validation_housing_tr = pd.DataFrame(
    x_validation_housing_tr,
    columns=x_validation_housing.columns,
    index=x_validation_housing.index
)

x_test_housing_tr = pd.DataFrame(
    x_test_housing_tr,
    columns=x_test_housing.columns,
    index=x_test_housing.index
)

# verifying if the imputation was successful
print("Null Data(Post Imputation): ")
print(x_train_housing_tr.isnull().sum())