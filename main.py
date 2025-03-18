import numpy as np
import csv
import helper as h
import matplotlib.pyplot as plt

data = h.loadWeatherData('weather_data.csv')
h.plothumidity(data[:,6])