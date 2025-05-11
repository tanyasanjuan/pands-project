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
df = df.dropna(how='all')


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
# The species are: setosa, versicolor, and virginica.
df['species'] = pd.Categorical(df['species'])
print(df['species'])


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


# Get the number of data for each species.
# The value_counts function returns the number of occurrences of each unique value in the column.
# Resource: https://www.geeksforgeeks.org/python-pandas-index-value_counts/
df['species'].value_counts()
print(df['species'].value_counts())


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

# Display the histogram of each variable.
# We can use histograms to visualize the distribution of a dataset.
# To display the histograms for each of the features, without having to duplicate the code 4 times, you can create for loops, which are useful for repeating an action several times.
# Resource: https://www.w3schools.com/python/python_for_loops.asp

# For loops will repeate the instruction, in this case for all the caracteristics of the iris flower.
# The for loop will iterate over the columns of the DataFrame, and for each column, it will create a histogram.
for column_name in df.columns[:-1]:  # Exclude the last column (species)
    # plt.figure(), creates a separe histogram per each characteristic. 
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
    plt.figure()
    plt.hist(df[column_name], edgecolor="black")
    plt.xlabel(column_name)
    plt.ylabel("frequency")
    plt.title(f"Histogram of {column_name}")
    # Show all histograms https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
    #plt.show()
    plt.savefig(f"{column_name}_histogram.png") # This will save the plots to a png file.

    # Explanation of each histogram:
    # Lenght sepal, 5.5 and 6.5 cm, is the most common size of the sepal. 5.8 cm is the mean.
    # Width sepal, 3 cm, is the most common size of the sepal. 3.0 cm is the mean.
    # Length petal, 1 and 5 cm, is the most common size of the petal. 3.7 cm is the mean.
    # Width petal, 0.1 and 1.5 cm, is the most common size of the petal. 1.2 cm is the mean.