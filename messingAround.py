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

def enlarge():
    directory = os.listdir("database")
    count = 0
    for file in directory:
        try:
            f = open("database/" + file, 'r')
            final_image = []
            if count <= 6:#top left
                for line in f:
                    temp = [int(x) for x in line.split(" ")]
                    for j in range(28):
                        temp.append(0)
                    final_image.append(temp)
                for q in range(28):
                    temp = [0] * 56
                    final_image.append(temp)
            if count > 6 and count <=12:#top right
                for line in f:
                    temp = [int(x) for x in line.split(" ")]
                    for j in range(28):
                        temp.insert(0,0)
                    final_image.append(temp)
                for q in range(28):
                    temp = [0] * 56
                    final_image.append(temp)
            if count > 12 and count <= 18:#bottom left
                for q in range(28):
                    temp = [0] * 56
                    final_image.append(temp)
                for line in f:
                    temp = [int(x) for x in line.split(" ")]
                    for j in range(28):
                        temp.append(0)
                    final_image.append(temp)
            if count > 18:#bottom right
                for q in range(28):
                    temp = [0] * 56
                    final_image.append(temp)
                for line in f:
                    temp = [int(x) for x in line.split(" ")]
                    for j in range(28):
                        temp.insert(0,0)
                    final_image.append(temp)
            final_file = open("larger_images/" + file, 'w')
            for j in range(len(final_image)):
                line = final_image[j]
                line_length = len(line)
                for x in range(len(line)):
                    final_file.write(str(line[x]) + " ") if x != line_length - 1 else final_file.write(str(line[x]))
                if j != len(final_image) - 1:
                    final_file.write("\n")
            final_file.close()
            f.close()
            count += 1
        except ValueError:
            print file

def enlarge_v_2():
    directory = os.listdir("category_database/queries")
    for files in directory:
        f = open("category_database/queries/" + files, 'r')
        r = random.randint(0,3)
        final_image = []
        if r == 0:
            for line in f:
                temp = [int(x) for x in line.split(" ")]
                for j in range(28):
                    temp.append(0)
                final_image.append(temp)
            for q in range(28):
                temp = [0] * 56
                final_image.append(temp)
        if r == 1:
            for line in f:
                temp = [int(x) for x in line.split(" ")]
                for j in range(28):
                    temp.insert(0, 0)
                final_image.append(temp)
            for q in range(28):
                temp = [0] * 56
                final_image.append(temp)
        if r == 2:
            for q in range(28):
                temp = [0] * 56
                final_image.append(temp)
            for line in f:
                temp = [int(x) for x in line.split(" ")]
                for j in range(28):
                    temp.append(0)
                final_image.append(temp)
        if r == 3:
            for q in range(28):
                temp = [0] * 56
                final_image.append(temp)
            for line in f:
                temp = [int(x) for x in line.split(" ")]
                for j in range(28):
                    temp.insert(0, 0)
                final_image.append(temp)
        final_file = open("larger_images/" + files, 'w')
        for j in range(len(final_image)):
            line = final_image[j]
            line_length = len(line)
            for x in range(len(line)):
                final_file.write(str(line[x]) + " ") if x != line_length - 1 else final_file.write(str(line[x]))
            if j != len(final_image) - 1:
                final_file.write("\n")
        final_file.close()
        f.close()


enlarge_v_2()