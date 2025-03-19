# Problem Overview
- This is an hour by hour weather data containing 7 different features that make up the weather condition. You need to forecast relative humidity of any particular time point given the weather data of past time points

# Dataset
- The dataset contains 7 weather features per time point representing the hourly climate data nearby Monash University (in Australia) from January 01, 2010 to May 31, 2021: total 100057 time points
- Feature details:
    - temperature: measured in celcius
    - wind speed: measured in (m/s)
    - mean sea level pressure: measured in (Pa)
    - surface solar radiation: measured in (W/m^2)
    - surface thermal radiation: measured in (W/m^2)
    - total cloud cover: ranges between 0 and 1
    - relative humidity: ranges between 0 and 1; this is the value you need to forecast (regression task)

# Task Description:
- You need to predict relative humidity of a time point based on data from previous time points without ever using relative humidity information from any time point
- We want you to try three configurations based on the usage of data from previous time points:
    (1) use past N (you can tune this) time point data starting from the immediate previous time point of the one to be forecasted for
        - you are trying to predict relative humidity for the next hour
    (2) use past N (you can tune this) time point data starting from the 6 time points prior to the one to be forecasted for
        - you are trying to predict what will happen to relative humidity after 6 hours from now
    (3) use past N (you can tune this) time point data starting from the 24 time points prior to the one to be forecasted for
        - you are trying to predict what will happen to relative humidity after one day
    Note:
    - You cannot use any weather information from the time point for which you are predicting humidity for in any of the above configurations
    - You cannot use relative humidity information from any time point

# Train-Test Data:
- Your training data should be completely disjoint from youre test data
- If you are using time point 1-50,000 in any of your training samples, then the test samples cannot belong to any of those time points
- My recommendation is to divide the data into 100 time point groups and randomly assign 80% of the groups to training and 20% to testing
- Perform cross validation on the training dataset to develop and choose your model; the test set should be kept only for testing the final model.

# Expectations:
- proper time series feature visualization
- feature selection attempts through correlation analysis
- You are expected to explore both deep learning based time series modeling and traditional ML techniques
