import numpy as np
import csv
import matplotlib.pyplot as plt
import os

def loadWeatherData(filename):
    # read in the weather CSV into a np array
    # return array
    rawlst = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter= ',')
        header = next(reader)
        for row in reader:
            rawlst.append(row)
    d = {}
    for i in range(0, len(header)):
        d[i] = header[i]
    arr = np.array(rawlst, dtype = float)
    return arr , d

def plotdata(data, col, d):
    y_val = data[:,col]
    name = d[col]
    time_steps = np.arange(len(data))
    plt.plot(time_steps, y_val, label = name, color = 'b', linestyle = '-')
    plt.xlabel("Time Steps")
    plt.ylabel(name)
    plt.title(f"Time Series Plot of {name}")

    save_folder = "plots"
    os.makedirs(save_folder, exist_ok=True)
    save_filename = f"{name}_plot.png"
    save_path = os.path.join(save_folder, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



    