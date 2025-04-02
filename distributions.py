# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# files
import helper as h 

# load data
df = pd.read_csv("weather_data.csv")

# remove outliers 
df = df[(df["relative_humidity"]<=1)]

# standardise data
scaler = StandardScaler()

# Create a categorical variable that classifies rows as Night and Day 
    # Define threshold: If surface solar radiation is greater than 0, classify as 'Day', else 'Night'
df["Day_or_Night"] = np.where(df["surface_solar_radiation"] > 0, "Day", "Night")

# Convert to categorical type (optional, for efficiency)
df["Day_or_Night"] = df["Day_or_Night"].astype("category")

# reorder the rows
cols = list(df.columns)
cols.insert(-1, cols.pop(cols.index("Day_or_Night")))
df = df[cols]

# View the changes
print(df.head())
