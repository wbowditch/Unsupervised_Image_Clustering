import numpy as np
import os
def _create_matrix(name):  # Reads image from textfile into numpy array
    file = open(name,'r')
    array = []
    for line in file:
        array.append([int(x) for x in line.split(" ")])
    return np.array(array)


database = []

for file in os.listdir("database"):
    print file
    x = _create_matrix("database/"+file)
    database.append(x)

print "row",max(database,key= lambda p : p.shape[0]).shape[0]

print "col",max(database,key= lambda p : p.shape[1]).shape[1]


