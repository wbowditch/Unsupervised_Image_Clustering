import scipy
import random
import os
def distort():
    #adds random 1s that distort the image throughout the array at a rate of 14% per point
    image_array = []
    for i in range(15):
        image = open("twos/two"+str(i)+".txt",'r')
        array = []
        for line in image:
            array.append([int(x) for x in line.split(" ")])
        image_array.append(array)

    for image in image_array:
        rows = len(image)
        columns = len(image[0])
        for i in range (rows):
            for j in range(columns):
                if image[i][j] != 1:
                    image[i][j] = 1 if random.randint(0,99) < 15 else 0
    for i in range(len(image_array)):
        file = open("category_database/test/newDirectory/blurredTwo.0." + str(i) + ".txt",'w')
        for line in image_array[i]:
            linelength = len(line)
            for j in range(linelength):
                file.write(str(line[j]))
                if j != linelength-1:
                    file.write(" ")
            file.write("\n")
        file.close()

distort()
