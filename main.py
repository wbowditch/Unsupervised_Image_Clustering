import sys
import os
from Image import *

"""  when you find the image and are trying ot scale it, you calculate the four corners and then find a buffer such that the array is square """


def main(argv):
    print argv
    database = []
    queries = []
    os.getcwd()

    for file in os.listdir("database"):
        if not file.startswith('.'):
            print file
            x = Image("database/"+file)
            #print x.shapes[0].shape_corners
            database.append(x)


    for file in os.listdir("queries"):
        if not file.startswith('.'):
            print file
            x = Image("queries/"+file)
            queries.append(x)

    #Decision tree - actual answer
    theta = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    area_clean = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    height_to_width = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    area_to_size = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    area_to_matrix = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    corner_count = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]} #number of corners before grouping

    for image in database:
        shape = image.shapes[0]
        if image.file_name.startswith('zero'):
            num = 0
        elif image.file_name.startswith('one'):
            num = 1
        elif image.file_name.startswith('two'):
            num = 2
        elif image.file_name.startswith('three'):
            num = 3
        elif image.file_name.startswith('four'):
            num = 4
        elif image.file_name.startswith('five'):
            num = 5
        elif image.file_name.startswith('six'):
            num = 6
        elif image.file_name.startswith('seven'):
            num = 7
        elif image.file_name.startswith('eight'):
            num = 8
        elif image.file_name.startswith('nine'):
            num = 9
        else:
            num = 10
        theta[num].append(float(shape.theta))
        area_clean[num].append(float(shape.area_clean))
        height_to_width[num].append(float(shape.height_to_width))
        area_to_size[num].append(float(shape.area_to_size))
        corner_count[num].append(float(len(shape.shape_corners)))
        area_to_matrix[num].append(float(shape.area_to_matrix))
    for num in range (0,10):
        print "For number " + str(num) + ":"
        print "Theta Average", float(sum(theta[num]))/float(len(theta[num]))
        print "Area Clean Average", float(sum(area_clean[num]))/float(len(area_clean[num]))
        print "Height Width Average", float(sum(height_to_width[num]))/float(len(height_to_width[num]))
        print "Area to Size Average", float(sum(area_to_size[num]))/float(len(area_to_size[num]))
        print "Corner Count Average", float(sum(corner_count[num]))/float(len(corner_count[num]))
        print "Area to Matrix Average",float(sum(area_to_matrix[num]))/float(len(area_to_matrix[num]))
        print

    for image in queries:
        print image.file_name
        #print "Name",image.file_name
        print image.decisionTree(database)

if __name__ == '__main__':
    main(sys.argv)