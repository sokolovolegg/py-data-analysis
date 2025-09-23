# imports

import pandas as pd
import math
import numpy as np
from datetime import datetime
import matplotlib as plt
# Task 1

"""
(a) Calculate the mean, uncertainty of the mean, variance, and standard deviation of the 
1. age distribution
2. total time distribution
directly from the data. (1 point)
Implement your own algorithm; you can compare your values to np.mean, np.var, np.std
"""

# Opening the file
df = pd.read_csv('ironman.txt', sep = '\t', header = None)
# to make it comfortable working with the file, i added the titles
df.columns = ['total rank',
              'year of birth', 
              'total time', 
              'swimming time', 
              'swimming rank', 
              'cycling time', 
              'cycling rank', 
              'running time', 
              'running rank'
              ]

print(df.head())

# Function mean
def mean_custom(values): 
    if len(values) == 0: 
        return None
    return sum(values) / len(values)

# Function variance
def variance_custom(values):
    n = len(values)
    if n == 0:
        return None
    return sum((x - mean_custom(values)) ** 2 for x in values) / (n-1)

# Function std 
def std_custom(values):
    if len(values) == 0:
        return None
    return math.sqrt(variance_custom(values))

# Function uncertainty of the mean 
def mean_uncertainty_custom(values):
    if len(values) == 0:
        return None
    return std_custom(values) / math.sqrt(len(values))

# Printing statistics for the year of birth
print(f"(a1) Mean age: {mean_custom(df['year of birth'])}")
print(f"(a2) Variance of age: {variance_custom(df['year of birth'])}")
print(f"(a3) Std of age: {std_custom(df['year of birth'])}")
print(f"(a4) Uncertainty of mean age: {mean_uncertainty_custom(df['year of birth'])}")
print("-" * 50) 
# Printing statistics for the total time
print(f"(b1) Mean total time: {mean_custom(df['total time'])}")
print(f"(b2) Variance of total time: {variance_custom(df['total time'])}")
print(f"(b3) Std of total time: {std_custom(df['total time'])}")
print(f"(b4) Uncertainty of mean total time: {mean_uncertainty_custom(df['total time'])}")
print("-" * 50)
# Printing statistics using NumPy
print("NumPy results:")
print(f"(a1) Mean age (NumPy): {np.mean(df['year of birth'])}")
print(f"(a2) Variance of age (NumPy): {np.var(df['year of birth'], ddof=1)}")
print(f"(a3) Std of age (NumPy): {np.std(df['year of birth'], ddof=1)}")
print(f"(a4) Uncertainty of mean age (NumPy): {np.std(df['year of birth'], ddof=1) / np.sqrt(len(df['year of birth']))}")
print("-" * 50)
print(f"(b1) Mean total time (NumPy): {np.mean(df['total time'])}")
print(f"(b2) Variance of total time (NumPy): {np.var(df['total time'], ddof=1)}")
print(f"(b3) Std of total time (NumPy): {np.std(df['total time'], ddof=1)}")
print(f"(b4) Uncertainty of mean total time (NumPy): {np.std(df['total time'], ddof=1) / np.sqrt(len(df['total time']))}")
print("-" * 50)

"""
(c) Compute the average of the total time for people younger and older than 35 years. Calculate the uncertainty on these two times. Can you conclude that one group is faster than the other? Argue! (2 points)
than the other? Argue! (2 points)
"""

year = datetime.now().year # I could also just pass 2025 here
df['age'] = year - df['year of birth'] 

# Splitting the data into two groups
younger_than_35 = df[df['age'] < 35] 
older_than_35 = df[df['age'] >= 35]

# Calculating means and uncertainties
mean_younger = mean_custom(younger_than_35['total time'])
mean_older = mean_custom(older_than_35['total time'])
uncertainty_younger = mean_uncertainty_custom(younger_than_35['total time'])
uncertainty_older = mean_uncertainty_custom(older_than_35['total time'])

print("(c) Results:")
print(f"Mean total time for younger than 35: {mean_younger} +- {uncertainty_younger}")
print(f"Mean total time for older than 35: {mean_older} +- {uncertainty_older}")
print("-" * 50)

"""
(d) Histogram and plot the age and total time distributions with error bars according to
the rule introduced in lecture 2. (4 points)
Implement your own algorithm. Hint: create a function for the binning, use plt.bar
for plotting; you can compare your values to np.histogram.
"""

