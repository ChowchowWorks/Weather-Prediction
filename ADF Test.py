import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load your dataset
filename = "weather_data.csv"
data = pd.read_csv(filename)

col_name = "relative_humidity"  # Change this to your target column
y_val = data[col_name].dropna()  # Remove missing values

# Perform the Augmented Dickey-Fuller test
adf_result = adfuller(y_val)

# Print results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])

# Check stationarity
if adf_result[1] < 0.05:
    print(f"{col_name} is stationary (Reject H0)")
else:
    print(f"{col_name} is NOT stationary (Fail to Reject H0)")
