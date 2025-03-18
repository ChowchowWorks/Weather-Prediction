import numpy as np
import csv

def loadWeatherData(filename):
    # read in the weather CSV into a np array
    # return array
    csv.lst = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter= ';')
        header = next(reader)
        for row in reader:
            csv.lst.append(row)
    arr = np.array(csv.lst)
    return arr

data = loadWeatherData('weather_data.csv')