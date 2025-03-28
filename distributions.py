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
data, d = h.loadWeatherData("weather_data.csv")
df = pd.read_csv("weather_data.csv")

# remove outliers 
data = h.removeOutliers(data)
df = df[(df["relative_humidity"]<=1)]

# standardise data
data = h.standardizer(data)
scaler = StandardScaler()
df[["temperature", 'wind_speed', 'mean_sea_level_pressure', 'surface_solar_radiation', 'surface_thermal_radiation', 'total_cloud_cover']] = scaler.fit_transform(df[["temperature", 'wind_speed', 'mean_sea_level_pressure', 'surface_solar_radiation', 'surface_thermal_radiation', 'total_cloud_cover']])



# temperature
temperature = data[:, 0]
#sns.histplot(np.log(temperature), bins= 30, kde=True)
plt.title("temperature histogram")
#plt.show()
#plt.close()
# wind_speed
wind = data[:, 1]
#sns.histplot(np.log(wind), bins = 30, kde=False)
plt.title("wind speed histogram")
#plt.show()
#plt.close()
# mean_sea_level_pressure
sea = data[:, 2]
#sns.histplot(sea, bins = 30, kde= True)
plt.title("sea level pressure histogram")
#plt.show()
#plt.close()

# surface_solar_radiation --- this is not normally distributed 
solar = data[:, 3]
#sns.histplot(solar, bins = 30, kde= True)
plt.title("surface solar radiation histogram")
#plt.show()
#plt.close()

# surface_thermal_radiaion
thermal = data[:, 4]
#sns.histplot(thermal, bins = 30, kde= True)
plt.title("surface thermal radiation histogram")
#plt.show()
#plt.close()

# total cloud cover -- this is not normally distributed
cloud = data[:, 5]
#sns.histplot(cloud, bins = 30, kde= True)
plt.title("total cloud cover histogram")
#plt.show()
#plt.close()

# correlation plots
sns.pairplot(df[["temperature", 'wind_speed', 'mean_sea_level_pressure', 'surface_solar_radiation', 'surface_thermal_radiation', 'total_cloud_cover']])
plt.title("Feature Pair Plots")
plt.show()
plt.close()

#contour plots
#sns.kdeplot(x= df["temperature"], y=df["surface_thermal_radiation"], fill=True)
plt.title("Contour plot: Temperature & Surface Thermal Radiation")
#plt.show()
#plt.close()

mi_temp_thermal = mutual_info_regression(df[["surface_solar_radiation"]], df["surface_thermal_radiation"])
#print(mi_temp_thermal)

x = 240
#plt.plot(np.arange(x),wind[:x])
#plt.plot(np.arange(x),temperature[:x])
plt.title("Temperature and Wind Speed")
#plt.show()
#plt.close()
