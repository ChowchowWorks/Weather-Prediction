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
scaler.fit_transform(df)

sns.histplot(df['temperature'], bins=45, kde=True)
plt.title("Histogram Plot of Temperature")
plt.show()
plt.close()