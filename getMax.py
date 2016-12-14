import numpy as np
import os
def _create_matrix(name):  # Reads image from textfile into numpy array
    file = open(name,'r')
    array = []
    for line in file:
        array.append([int(x) for x in line.split(" ")])
    return np.array(array)


database = []

for file in os.listdir("queries"):
    if not file.startswith('.'):
        print file
        x = _create_matrix("queries/"+file)
        database.append([x,file])


def cleanMatrix(array):
    clean_matrix = np.zeros((280,280)).astype(int)
    for r in range(array.shape[0]):
        for c in range(array.shape[1]):
            clean_matrix[r][c] = array[r][c]
    return clean_matrix

for i in range(len(database)):
    database[i][0] = cleanMatrix(database[i][0])

for pic in database:
    array,file_name = pic
    print array.shape
    file = open("database/"+file_name,'w')
    file.seek(0)
    file.truncate()
    for row in array:
        line1 = [str(int(x)) for x in row]
        line = ' '.join(line1)
        file.write(line + '\n')
    file.close()




#print "row",max(database,key= lambda p : p.shape[0]).shape[0]

#print "col",max(database,key= lambda p : p.shape[1]).shape[1]


