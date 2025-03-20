import numpy as np
import csv
import helper as h
import matplotlib.pyplot as plt

data, d = h.loadWeatherData('weather_data.csv')
h.plotdata(data, 1, d)