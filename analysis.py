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
# Meaning it make the connection to a web server, to have a file.
# Pandas helps to interpretate the file received.
