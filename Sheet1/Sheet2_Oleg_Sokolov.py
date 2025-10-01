# imports

import pandas as pd
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

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
    if n < 2:
        return 0.0
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

# function for binning and creating histograms
def histogram_custom(data, step=None, bins=None):
    data = np.asarray(data)
    # creating edges
    if bins is not None:
        edges = np.asarray(bins, dtype=float)
    elif step is not None:
        data_min, data_max = data.min(), data.max()
        start = math.floor(data_min / step) * step
        end = math.ceil(data_max / step) * step
        edges = np.arange(start, end + step, step, dtype=float)
        if edges[-1] < data_max:
            edges = np.append(edges, edges[-1] + step)
    else:
        raise ValueError("Either step or bins must be provided")
    
    # counting counts
    k = len(edges) - 1
    counts = np.zeros(k, dtype=int)
    for v in data:
        idx = np.searchsorted(edges, v, side="right") - 1
        if idx < 0:
            idx = 0
        if idx >= k:
            idx = k - 1
        counts[idx] += 1
    return counts, edges

# plotting with errors
def histogram_with_errors_custom(counts, edges, title="", ax=None):
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    errors = np.sqrt(counts)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)) 
    ax.bar(centers, counts, width=widths, align="center", alpha=0.6)
    ax.errorbar(centers, counts, yerr=errors, fmt="none", ecolor="k",
                capsize=3, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Counts")
    ax.set_xticks(edges)
    ax.grid(axis="y", alpha=0.2)
    return ax

# age
age = df["age"].dropna()
age_counts, age_edges = histogram_custom(age, step=5)
print("Age bin edges:", age_edges)
print("Age counts:", age_counts)

# check with numpy
np_counts, np_edges = np.histogram(age, bins=age_edges)
print("np.histogram counts equal:", np.array_equal(np_counts, age_counts))

ax1 = histogram_with_errors_custom(age_counts, age_edges,
                            title="Age distribution (step=5 years)")

# total time
tt = df["total time"].dropna().astype(float)
n_tt = len(tt)
sturges_k = int(np.ceil(1 + np.log2(n_tt))) if n_tt > 1 else 1
tt_edges = np.linspace(tt.min(), tt.max(), sturges_k + 1)
tt_counts, tt_edges = histogram_custom(tt, bins=tt_edges)

# check with numpy
np_counts_tt, np_edges_tt = np.histogram(tt, bins=tt_edges)
print("np.histogram counts equal (total time):", np.array_equal(np_counts_tt, tt_counts))

ax2 = histogram_with_errors_custom(tt_counts, tt_edges,
                            title=f"Total time distribution (Sturges k={sturges_k})")

# plt.tight_layout()
# plt.show()

""" (e) Calculate the mean, variance and standard deviation of the distribution from the histograms of the age and total time. Compare the results with those you got in a). Use
different bin widths and comment on what you observe. (2 points)
Hint: if you did not implement the histogram function flexible enough above, you can
also use np.histogram to create a histogram here.
"""

def statistics_from_histogram(counts, edges):
    counts = np.asarray(counts)
    edges = np.asarray(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    total = counts.sum()
    mean = np.sum(counts * centers) / total
    variance = np.sum(counts * (centers - mean)**2) / (total - 1)
    std = np.sqrt(variance)
    return mean, variance, std

# age
mean_age_hist, var_age_hist, std_age_hist = statistics_from_histogram(age_counts, age_edges)
print(f"Age from histogram: mean = {mean_age_hist:.2f}, var = {var_age_hist:.2f}, std = {std_age_hist:.2f}")

# total time
mean_tt_hist, var_tt_hist, std_tt_hist = statistics_from_histogram(tt_counts, tt_edges)
print(f"Total time from histogram: mean = {mean_tt_hist:.2f}, var = {var_tt_hist:.2f}, std = {std_tt_hist:.2f}")


""" (f) Calculate the covariance and correlation coefficient, using the formulae introduced in
lecture 3, between
• the total rank and the total time
• year of birth and the total time
• the total time and the swimming time
• the cycling time and the running time.
Take a look at the scatter plots from exercise sheet no. 1 and compare the results with
the results you got. (4 points)
Convert the total time from minutes to seconds and calculate the covariance and correlation coefficient between the age and the total time again. Which of the two changes
and which stays the same? (1 point)
"""

def covariance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / (n - 1)

def correlation(x, y):
    return covariance(x, y) / (np.std(x, ddof=1) * np.std(y, ddof=1))

# 1. total rank and total time
cov_rank_tt = covariance(df["total rank"], df["total time"])
corr_rank_tt = correlation(df["total rank"], df["total time"])

# 2. year of birth and total time
cov_yob_tt = covariance(df["year of birth"], df["total time"])
corr_yob_tt = correlation(df["year of birth"], df["total time"])

# 3. total time and swimming time
cov_tt_swim = covariance(df["total time"], df["swimming time"])
corr_tt_swim = correlation(df["total time"], df["swimming time"])

# 4. cycling time and running time
cov_cycl_run = covariance(df["cycling time"], df["running time"])
corr_cycl_run = correlation(df["cycling time"], df["running time"])

print("Covariances:")
print(f"Total rank & Total time: {cov_rank_tt:.2f}")
print(f"Year of birth & Total time: {cov_yob_tt:.2f}")
print(f"Total time & Swimming time: {cov_tt_swim:.2f}")
print(f"Cycling time & Running time: {cov_cycl_run:.2f}")

print("\nCorrelation coefficients:")
print(f"Total rank & Total time: {corr_rank_tt:.4f}")
print(f"Year of birth & Total time: {corr_yob_tt:.4f}")
print(f"Total time & Swimming time: {corr_tt_swim:.4f}")
print(f"Cycling time & Running time: {corr_cycl_run:.4f}")

total_time_sec = df["total time"] * 60  # in seconds

# covariance and correlation betwen age and total time in seconds
cov_age_tt_sec = covariance(df["age"], total_time_sec)
corr_age_tt_sec = correlation(df["age"], total_time_sec)

print("\nAge & Total time (seconds):")
print(f"Covariance: {cov_age_tt_sec:.2f}")  # will change
print(f"Correlation: {corr_age_tt_sec:.4f}")  # will not change


# Task 2 
""" 
You have been hired as a medical physicist, and your first task is to report on the level of
radiation in the room where the containers of some radioactive sources, used for radiotherapy
research, are stored.
You measure the data stored in radiation.txt which can be found on the webpage. The
measurements are taken with different dosimeters, and under different conditions, and thus
have different uncertainties (2nd column). Assume that all measurements are independent
from each other. The level of radiation is measured in mSv/h, millisievert per hour1.
"""

"""
(a) Calculate the average radiation level, in mSv per year, and the associated uncertainty
using the relevant formulae introduced in lecture 3. (3 points)
"""

# importing the data
df_rad = pd.read_csv('radiation.txt', header=None, sep='\s+')
df_rad.columns = ['value', 'uncertainty']

# assigning names to columns
values = df_rad['value'].to_numpy()
uncertainties = df_rad['uncertainty'].to_numpy()

weights = 1 / uncertainties**2
weighted_mean = np.sum(weights * values) / np.sum(weights)
weighted_uncertainty = np.sqrt(1 / np.sum(weights))

"""
(b) Based on your result, argue and explain whether the level of radiation in that room is
compatible with the natural background radiation, which has been measured to be 2.4
mSv/y. You can assume that the uncertainty on this value is negligibly small compared
to the uncertainty on your measurements. (1 point)
"""

# converting to mSv/year
hours_per_year = 24 * 365
mean_year = weighted_mean * hours_per_year
unc_year = weighted_uncertainty * hours_per_year


print(f"Mean level of radiation: {mean_year:.2f} ± {unc_year:.2f} mSv/year")

# Comparing with natural radiation
natural_background = 2.4
diff = mean_year - natural_background
sigma = diff / unc_year

print(f"Difference with natural background: {diff:.2f} mSv/year (~{sigma:.1f} σ)")
if abs(sigma) <= 2:
    print("Radiation level is compatible with natural background")
else:
    print("Radiation level is above natural background and statistically significant")



