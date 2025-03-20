import numpy as np
import csv
import helper as h
import matplotlib.pyplot as plt

data, d = h.loadWeatherData('weather_data.csv')
for i in range(data.shape[1]):
    h.plotdata(data, i, d)