import csv
import numpy as np
#define function to read csv data into an array given a filename
def opencsv(file):
    with open("".join( ["data/",file] ), 'r') as f:
        r = csv.reader(f)
        data = list(r)
    #return array of values
    return np.array(data,dtype=float)