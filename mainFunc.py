# Needed imports 
from sklearn.datasets import load_iris
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
  
# Fetch dataset 
#iris = fetch_ucirepo(id=53) 
iris = load_iris()

# Create the pandas DataFrames
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
target_df = pd.DataFrame(data = iris.target, columns = ['species'])

# Generate the labels 
def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virginica'

# Apply the converter
target_df['species'] = target_df['species'].apply(converter)

# Concatenate the DataFrames
df = pd.concat([iris_df, target_df], axis = 1)

# Plot the data // Uncomment to see the info about what is in the data set (sepal length, sepal width, petal length, petal width)
#df.info()

# Display random data samples // Uncomment to see the random data samples
#df.sample(10)

# Display the data columns // Uncomment to see the data columns
#df.columns

# Compute the correlation coefficient for iris data // Uncomment to see the correlation coefficient
#df.corr()

# Displaying the correlation coefficient as a heatmap // Uncomment to see the heatmap

# Visualize iris features as a heatmap
'''
numeric_cols_df = df.select_dtypes(include=[np.number])
cor_eff = numeric_cols_df.corr(method='pearson')
plt.figure(figsize=(6, 6))
sns.heatmap(cor_eff, linecolor = 'white', linewidths = 1, annot = True)

# Plot the lower half of the coorelation matrix
fig, ax = plt.subplots(figsize=(6, 6))

# Compute the correlation matrix
mask = np.zeros_like(cor_eff) 

mask[np.triu_indices_from(mask)] = 1
sns.heatmap(cor_eff, linecolor = 'white', linewidths = 1, mask = mask, ax = ax, annot = True) 
'''




# Visualize the data using pairplot // Uncomment to see the pairplot
g = sns.pairplot(df, hue = 'species') 

# Display the graph 
plt.show() 

#print(df) 
  
# Data (as pandas dataframes) 
#X = iris.data.features 
#y = iris.data.targets 
  
# Metadata 
#print(iris.metadata) 
  
# Variable information 
#print(iris.variables) 
