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
df = pd.read_csv(filename, header=None)


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
# The other columns are the features.
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
'''
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
    # Save the histogram to a png file.
    # plt.savefig(f"{column_name}_histogram.png") 
    
    # Show all histograms https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
    #plt.show()
    
'''
# Explanation of each histogram:
# Lenght sepal, 5.5 and 6.5 cm, is the most common size of the sepal. 5.8 cm is the mean.
# Width sepal, 3 cm, is the most common size of the sepal. 3.0 cm is the mean.
# Length petal, 1 and 5 cm, is the most common size of the petal. 3.7 cm is the mean.
# Width petal, 0.1 and 1.5 cm, is the most common size of the petal. 1.2 cm is the mean.

# We can also check the relationship between the features.
# We can use scatter plots to visualize the relationship between two variables.
# To have more than two scatter plots, we can use subplots.
# Subplots are used to create multiple plots in a single figure.
# Resource: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
# https://interactivechaos.com/en/node/605


# import numpy library.
import numpy as np

colors = {
    'Iris-setosa': 'salmon',
    'Iris-versicolor': 'turquoise',
    'Iris-virginica': 'violet'
}
# With ".unique()" we extract the three unique species: 'setosa', 'versicolor', 'virginica'
species = df['species'].unique()

y = df['sepal length (cm)']
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
ax[0, 0].scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=[colors[s] for s in df['species']])
ax[0, 0].set_xlabel('sepal length (cm)')
ax[0, 0].set_ylabel('sepal width (cm)')
ax[0, 0].set_title('Sepal Length vs Sepal Width')
ax[0, 1].scatter(df['petal length (cm)'], df['petal width (cm)'], c=[colors[s] for s in df['species']])
ax[0, 1].set_xlabel('petal length (cm)')
ax[0, 1].set_ylabel('petal width (cm)')
ax[0, 1].set_title('Petal Length vs Petal Width')
ax[1, 0].scatter(df['sepal length (cm)'], df['petal length (cm)'], c=[colors[s] for s in df['species']])
ax[1, 0].set_xlabel('sepal length (cm)')
ax[1, 0].set_ylabel('petal length (cm)')
ax[1, 0].set_title('Sepal Length vs Petal Length')
ax[1, 1].scatter(df['sepal width (cm)'], df['petal width (cm)'], c=[colors[s] for s in df['species']])
ax[1, 1].set_xlabel('sepal width (cm)')
ax[1, 1].set_ylabel('petal width (cm)')
ax[1, 1].set_title('Sepal Width vs Petal Width')
ax[2, 0].scatter(df['sepal length (cm)'], df['petal width (cm)'], c=[colors[s] for s in df['species']])
ax[2, 0].set_xlabel('sepal length (cm)')
ax[2, 0].set_ylabel('petal width (cm)')
ax[2, 0].set_title('Sepal Length vs Petal Width')
ax[2, 1].scatter(df['sepal width (cm)'], df['petal length (cm)'], c=[colors[s] for s in df['species']])
ax[2, 1].set_xlabel('sepal width (cm)')
ax[2, 1].set_ylabel('petal length (cm)')
ax[2, 1].set_title('Sepal Width vs Petal Length')

# Create custom legend handles to put only one legend above all subplots
# Resource: https://chatgpt.com/share/6821e2e7-e3d8-8012-8750-d37fe43e1460
from matplotlib.patches import Patch
legend_handles = [Patch(color=color, label=label) for label, color in colors.items()]

# Add a single legend above all subplots
fig.legend(handles=legend_handles, loc='upper center', ncol=len(colors), bbox_to_anchor=(0.5, 1.02))

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0, 1, 0.97])

# plt.tight_layout() avoid overlapping of the plots.
# Resource: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html
plt.tight_layout()
# Save the scatter plot to a png file.
#plt.savefig("scatter__investigate_relationships.png")
#plt.show()



# explanation of the scatter plot:
# The scatter plot shows the relationship between the sepal length and petal length of the three species.
# The setosa species has a smaller sepal length and petal length than the other two species, and this is the most common size of the sepal and petal.
# The setosa has a smaller petal width than the other two species.
# The versicolor species has a larger sepal length and petal length than the setosa species, but smaller than the virginica species.
# Virginica species has the largest sepal and petal length, applying the same for the width.
# The versicolor species has a larger sepal width and petal width than the setosa species, but smaller than the virginica species.
# The petal length and sepal length are positively correlated, meaning that as one increases, the other also increases.
# And it's the same for the petal and sepal width.


# For better visualization of the data, we can use a box plot.
# Boxplot of the petal lengths for each species.
# https://www.w3schools.com/python/python_lists_comprehension.asp

plt.figure(figsize=(8, 6))

iris_data_boxplot = [df[df['species'] == sp_name]['petal length (cm)'] for sp_name in species]

# Boxplot and style
plt.boxplot(iris_data_boxplot, tick_labels=species)
plt.grid(axis='y')

# labels y title
plt.xlabel('Species')
plt.ylabel('Petal length (cm)')
plt.title('Boxplot of Petal Lengths')
plt.savefig("petal_lengths_boxplot.png")
plt.show()



# Boxplot of the petal width for each species.
plt.figure(figsize=(8, 6))
iris_data_boxplot = [df[df['species'] == sp_name]['petal width (cm)'] for sp_name in species]

# Boxplot and style
plt.boxplot(iris_data_boxplot, tick_labels=species)
plt.grid(axis='y')

# labels y title
plt.xlabel('Species')
plt.ylabel('Petal width (cm)')
plt.title('Boxplot of Petal Widths')
plt.savefig("petal_widths_boxplot.png")
plt.show()



# To save the summary of each variable to a text file.
summary = df.describe()
with open('iris_summary.txt', 'w') as f:
    f.write("Iris Dataset Summary\n")
    f.write(str(summary))