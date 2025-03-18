import numpy as np
import csv
import matplotlib.pyplot as plt

def loadWeatherData(filename):
    # read in the weather CSV into a np array
    # return array
    csv.lst = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter= ';')
        header = next(reader)
        print(header)
        for row in reader:
            csv.lst.append(row)
    arr = np.array(csv.lst)
    return arr

def plothumidity(data):
    y_val = data
    time_steps = np.arange(len(data))
    plt.plot(time_steps, y_val, label = 'reletive humidity', color = 'b', linestyle = '-')
    plt.xlabel("Time Steps")
    plt.ylabel("Humidity")
    plt.title("Time Series Plot of Reletive Humidity")
