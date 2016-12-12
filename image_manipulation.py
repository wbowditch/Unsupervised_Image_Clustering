import numpy as np
import os
def distort(n):
    #adds random 1s that distort the image throughout the array at a rate of 14% per point
    os.rmdir("category_database/blurredThrees")
    image_array = []
    for i in range(n):
        image = open("category_database/threes/three"+str(i)+".txt",'r')
        array = []
        for line in image:
            array.append([int(x) for x in line.split(" ")])
        image_array.append(array)
    for j in range(len(image_array)):
        if np.random.randint(0,99) < 14:
            image = image_array[j]
            rows = len(image)
            columns = len(image[0])
            for i in range (rows):
                for j in range(columns):
                    if image[i][j] != 1:
                        image[i][j] = 1 if np.random.randint(0,99) < 15 else 0
    for i in range(len(image_array)):
        file = open("category_database/blurredThrees/randomlyBlurredThrees.0." + str(i) + ".txt",'w')
        for line in image_array[i]:
            linelength = len(line)
            for j in range(linelength):
                file.write(str(line[j]))
                if j != linelength-1:
                    file.write(" ")
            file.write("\n")
        file.close()