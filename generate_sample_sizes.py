import numpy as np
import os
def generate(n):
    #adds random 1s that distort the image throughout the array at a rate of 14% per point
    os.rmdir("category_database/blurredThrees")
    for i in range(n):
        image = open("category_database/threes/three"+str(i)+".txt",'r')
        file = open("category_database/blurredThrees/randomlyBlurredThrees.0." + str(i) + ".txt",'w')
        array = []
        for line in image:
            array.append([int(x) for x in line.split(" ")])

    for i in range(len(image_array)):
        for line in image_array[i]:
            linelength = len(line)
            for j in range(linelength):
                file.write(str(line[j]))
                if j != linelength-1:
                    file.write(" ")
            file.write("\n")
        file.close()