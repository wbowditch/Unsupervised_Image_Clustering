import sys
import os
from Image import *

"""  when you find the image and are trying ot scale it, you calculate the four corners and then find a buffer such that the array is square """


def main(argv):
    print argv
    database = []
    queries = []
    os.getcwd()

    x = Image("multiple_shape_images/rat08.txt")
    queries.append(x)

    #x = Image("advanced_test/database/children19.txt")
    # file = open("advanced_test/database/rat08.txt",'r')
    # for row in file:
    #     queries.append([int(x) for x in row.split(' ')])
    # length = len(zip(*queries))
    # for line in queries:
    #     if len(line) < length:
    #         while len(line) != length:
    #             line.append(0)
    #     if len(line) > length:
    #         while len(line) != length:
    #             del line[-1]
    # file = open("multiple_shape_images/rat08.txt", 'w')
    # for line in range(len(queries)):
    #     file.write(' '.join(map(str,queries[line])))
    #     if line != len(queries) - 1:
    #         file.write("\n")
    # file.close()
    # add_image = open("advanced_test/database/children19.txt",'r')
    # for row in add_image:
    #     file.write(row)
    # for file in os.listdir("advanced_test/queries"):
    #     for row in file:
    #         row =

    # for row in range(x.original_matrix.shape[0]):
    #     for col in range(int(x.original_matrix.shape[1]/2)):
    #         x.original_matrix[-row, -col] = x.original_matrix[row,col]

    # image = open("advanced_test/database/children18.txt", 'w')
    # for row in range(x.original_matrix.shape[0]):
    #     image.write(' '.join(map(str,x.original_matrix[row])))
    #     if row != x.original_matrix.shape[0]-1:
    #         image.write("\n")
    # image.close()



    for image in queries:
        i = 0
        for shape in image.shapes:
            f = open("process_steps/" + str(i) +"_" + image.file_name[:-3] + "_clean.txt", 'w')
            for row in shape.clean_matrix:
                f.write(' '.join(map(str, row)))
                f.write("\n")
            f.close()
            f = open("process_steps/" + str(i) +"_" + image.file_name[:-3] + "_centered.txt", 'w')
            for row in shape.centered_matrix:
                f.write(' '.join(map(str, row)))
                f.write("\n")
            f.close()
            f = open("process_steps/" + str(i) +"_" + image.file_name[:-3] + "_rotated.txt", 'w')
            for row in shape.rotated_matrix:
                f.write(' '.join(map(str, row)))
                f.write("\n")

            f.close()
            f = open("process_steps/" + str(i) + "_" + image.file_name[:-3] + "_zoomed.txt", 'w')
            for row in shape.zoom():
                f.write(' '.join(map(str, row)))
                f.write("\n")
            f.close()
            f = open("process_steps/" + str(i) +"_" + image.file_name[:-3] + "_scaled.txt", 'w')
            for row in shape.scaled_matrix:
                f.write(' '.join(map(str, row)))
                f.write("\n")
            f.close()
            i+=1



if __name__ == '__main__':
    main(sys.argv)