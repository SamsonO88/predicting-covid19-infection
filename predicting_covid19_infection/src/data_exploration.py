# import libraries for exploratory analysis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv(r"C:\Users\Dell\Documents\my_linux\repos_\predicting-covid19-infection\predicting_covid19_infection\data\raw\COVID19 - COVID19.csv")
df.shape

# general summary statistics
df.describe(exclude="number")

# general summary statistics
df.describe()

# summary statistics for male
df[df["Sex"] == "MALE"].describe()

# summary statistics for female
df[df["Sex"] == "FEMALE"].describe()

# Create a FacetGrid with multiple charts (one per category in "sex")
g = sns.FacetGrid(df, col="Sex", height=5, aspect=1)
g.map(sns.histplot, "age", bins=5, kde=True)  # Use a histogram for age distribution
plt.title("Distrubtion of age across the different sex")
plt.show()

g = sns.FacetGrid(df, col="Result", height=5, aspect=1)
g.map(sns.histplot, "age", bins=5, kde=True)  # Use a histogram for age distribution
plt.title("Distrubtion of age across the different result")
plt.show()

# Create a cross-tabulation
crosstab = pd.crosstab(df['Sex'], df['Result'])

# Plot as heatmap
plt.figure(figsize=(5, 3))
sns.heatmap(crosstab, annot=True, cmap="coolwarm", fmt="d")
plt.title("Heatmap of Sex vs. Covid Result")
plt.show()

# Count comparison of two categorical variables
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sex', hue='Result')  # Compare 'sex' against 'covid result'
plt.title("Comparison of Sex and Smoking Habit")
plt.show()

# Count comparison of two categorical variables
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Sex', y='age', hue='Result')  # Compare 'sex' against 'smoker'
plt.title("Comparison of Sex and Covid result")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Sex', y='age', hue='Result')
plt.title("Age Distribution by Sex and Covid Result")
plt.show()