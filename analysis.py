# This program analyzes the Iris dataset.
# It loads the dataset, performs exploratory data analysis (EDA), and visualizes the data.
# Outputs a summary of each variable to a single text file.
# Saves a histogram of each variable to png files, and
# Outputs a scatter plot of each pair of variables.
# Author: Tanya San Juan.

# Import the libraries we need for data analysis.

# Pandas is a data analysis library for Python.
import pandas as pd

# Scikit-learn is a machine learning library for Python. 
# It contains datasets examples, including the Iris dataset.
import sklearn as skl

# Resources: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# To read the dataset, we use the read_csv function from the pandas library.
# The dataset is in CSV format, and this specify the path to the file. 
# Meaning, it make the connection to a web server, to have a file.
# Pandas helps to interpretate the file received.


# Import the Iris dataset.
# Resources: https://archive.ics.uci.edu/dataset/53/iris
filename = "iris_data.csv"

# The read_csv function reads the CSV file and stores it in a DataFrame (df).
# A DataFrame is a two-dimensional data structure with rows and columns, like a table or array.
# Resource: https://www.w3schools.com/python/pandas/pandas_dataframes.asp
# Read the file
df = pd.read_csv(filename)

# Add column names to the DataFrame
# Four columnns that are the caracteristics of the iris flower.
# The four feature names, are independent variables, which are characteristics measured in centimeters, 
# related to the sepals and petals of each species.
# Source: https://www.geeksforgeeks.org/iris-dataset/
df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']

print(df)
# With print (df), we can see its a data set with 150 rows and 5 columns.
# we can see the first 5 rows of the dataset.
# The first row is the header, which contains the names of the columns.
# The first column is the index, which is a unique identifier for each row.
# The index is not part of the dataset, but it's used to identify each row.
# The 5th column is the species of the iris flower.
# The target variable is the variable we want to predict.
# The other columns are the features, 
# which are the variables we use to predict the target variable.
# The last 5 rows of the dataset are the same as the first 5 rows. 

# Display the species of the iris flower.
df['species'] = pd.Categorical(df['species'])
print(df['species'])
# The species column is a categorical variable.

# To get the first 5 rows of the dataset, we can use the head function.
# The head function returns the first 5 rows of the dataset.
df.head()
print(df.head())
# The first 5 rows in the dataset are showing the characteristics of the specie setosa, 
# including the: sepal_length,sepal_width, petal_length, petal_width

# To print the last 5 rows of the dataset, we can use the pandas module and the "tail" 
# method to print the last rows starting from the bottom of the dataset.
# Resource: https://www.w3schools.com/python/pandas/pandas_analyzing.asp
df.tail()
print(df.tail())
# The last 5 rows in the dataset are showing the characteristics of the specie virginica, 
# including the: sepal_length,sepal_width, petal_length, petal_width

# To have more information about the dataset, we can use the describe function.
# Describe the dataset
df.describe()
print(df.describe())
# The describe() function returns a summary of the dataset:
# - count: Number of non-null values in the column.
# - mean: Average value of the column.
# - std: Standard deviation of the column, which measures the amount of variation or dispersion in the data.
# - min: The minimum value in the column.
# - 25%: The first quartile, which is the value below which 25% of the data falls.
# - 50%: The median, which is the value below which 50% of the data falls.
# - 75%: The third quartile, which is the value below which 75% of the data falls.
# - max: The maximum value in the column.
# Resource for quantiles: https://en.wikipedia.org/wiki/Quantile
# Resource for quartiles: https://en.wikipedia.org/wiki/Quartile

# To visualize the data, we can use the matplotlib library.
# Matplotlib is a plotting library for Python.
import matplotlib.pyplot as plt

