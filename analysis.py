# This program analyzes the Iris dataset.
# It loads the dataset, performs exploratory data analysis (EDA), and visualizes the data.
# Outputs a summary of each variable to a single text file.
# Saves a histogram of each variable to png files, and
# Outputs a scatter plot of each pair of variables.
# Author: Tanya San Juan.

# Import the libraries we need for data analysis.
import pandas as pd

import sklearn as skl

# Import the Iris dataset.
# Resources: https://archive.ics.uci.edu/dataset/53/iris
filename = "iris_data.csv"

# To read the dataset, we use the read_csv function from pandas library.
# The read_csv function reads the CSV file and stores it in a DataFrame (df).
# Resources: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# Resource: https://www.w3schools.com/python/pandas/pandas_dataframes.asp
# Read the file
df = pd.read_csv(filename, header=None)


# Add column names to the DataFrame
# Resource: https://www.geeksforgeeks.org/iris-dataset/
df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
print(df) 

# Display the species of the iris flower.
# The species are: setosa, versicolor, and virginica.
df['species'] = pd.Categorical(df['species'])
print(df['species'])


# To get the first 5 rows of the dataset, we can use the head function.
# The head function returns the first 5 rows of the dataset.
df.head()
print(df.head())


# To print the last 5 rows of the dataset, we can use the pandas module and the "tail" 
# method to print the last rows starting from the bottom of the dataset.
# Resource: https://www.w3schools.com/python/pandas/pandas_analyzing.asp
df.tail()
print(df.tail())


# Get the number of data for each species.
# The value_counts function returns the number of occurrences of each unique value in the column.
# Resource: https://www.geeksforgeeks.org/python-pandas-index-value_counts/
df['species'].value_counts()
print(df['species'].value_counts())


# To have more information about the dataset, we can use the describe function.
# Describe the dataset
df.describe()
print(df.describe())
# The describe() function returns a summary of the dataset.
# The summary includes the count, mean, standard deviation, minimum, maximum, and quartiles of each variable.


# To visualize the data, we can use the matplotlib library.
# Matplotlib is a plotting library for Python.
import matplotlib.pyplot as plt

# Display the histogram of each variable.
# Resource: https://www.w3schools.com/python/python_for_loops.asp

for column_name in df.columns[:-1]:  # Exclude the last column (species)
    # plt.figure(), creates a separe histogram per each characteristic. 
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
    plt.figure()
    plt.hist(df[column_name], edgecolor="black")
    plt.xlabel(column_name)
    plt.ylabel("frequency")
    plt.title(f"Histogram of {column_name}")
    # Save the histogram to a png file.
    plt.savefig(f"{column_name}_histogram.png") 

    # Show all histograms 
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
    plt.show()    


# we can use scatter plots to visualize the relationship between two variables.
# To have more than two scatter plots, we can use subplots.
# Subplots are used to create multiple plots in a single figure.
# Resource: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
# https://interactivechaos.com/en/node/605

# Check the relationship between the features.
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
plt.savefig("scatter__investigate_relationships.png")
plt.show()


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


# Boxplot of the sepal length for each species.
plt.figure(figsize=(8, 6))
iris_data_boxplot = [df[df['species'] == sp_name]['sepal length (cm)'] for sp_name in species]

# Boxplot and style
plt.boxplot(iris_data_boxplot, tick_labels=species)
plt.grid(axis='y')

# labels y title
plt.xlabel('Species')
plt.ylabel('Sepal length (cm)')
plt.title('Boxplot of Sepal length')
plt.savefig("sepal_lengths_boxplot.png")
plt.show()


# Boxplot of the sepal length for each species.
plt.figure(figsize=(8, 6))
iris_data_boxplot = [df[df['species'] == sp_name]['sepal width (cm)'] for sp_name in species]

# Boxplot and style
plt.boxplot(iris_data_boxplot, tick_labels=species)
plt.grid(axis='y')

# labels y title
plt.xlabel('Species')
plt.ylabel('Sepal width (cm)')
plt.title('Boxplot of Sepal width')
plt.savefig("sepal_widths_boxplot.png")
plt.show()


# To display the heatmap we can use seaborn.
# Import seaborn. https://seaborn.pydata.org/generated/seaborn.heatmap.html
import seaborn as sns
import matplotlib.pyplot as plt

# Figsize creat a Matplotlib figure with a custom size.
# in this case figsize = width of 8, and 6 of height.
plt.figure(figsize=(8, 6))

# Resource: https://chatgpt.com/share/68224907-43e4-8012-aece-0d4afd1774ac 
# correlation_matrix - contains the values ​​between -1 and 1.
correlation_matrix = df.corr(numeric_only=True)
# annot=True: Draws numerical values ​​directly within each map cell.
# cmap= selected the color by name,in this case 'flare'
# https://seaborn.pydata.org/generated/seaborn.color_palette.html
# fmt= is the format of the numbers, represented by 2 decimals. 

sns.heatmap(correlation_matrix, annot=True, cmap='flare', fmt=".2f")
plt.title("Heatmap Correlation Matrix")
plt.savefig("heatmap_correlation_matrix.png")
plt.show()


# To save the summary of each variable to a text file.
summary = df.describe()
with open('iris_summary.txt', 'w') as f:
    f.write("Iris Dataset Summary\n")
    f.write(str(summary))