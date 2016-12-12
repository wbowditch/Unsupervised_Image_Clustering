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
        print file
        x = Image("database/" + file)
        database.append(x)

    for file in os.listdir("queries"):
        print file
        x = Image("queries/" + file)
        queries.append(x)

    for image in queries:
        print "Name", image.file_name
        print image.s_z_r_b_matrix
        print "Area", image.area_
        print "b_Area", image.b_area_
        print "Center", image.b_center
        print "Angle", image.b_radians
        print "Corner Count", len(image.corners)
        print "Corners Group Count", len(image.grouped_corners)
        print "Corner Groups", image.grouped_corners
        print "Distnaces:" + "\n", image.distances()
        print "Edge List: ", image.edgelist
        print
        print

    for image in database:
        print "Name", image.file_name
        print image.s_z_r_b_matrix
        print "Area", image.area_
        print "b_Area", image.b_area_
        print "Center", image.b_center
        print "Angle", image.b_radians
        print "Corner Count", len(image.corners)
        print "Corners Group Count", len(image.grouped_corners)
        print "Corner Groups", image.grouped_corners
        print "Distnaces:" + "\n", image.distances()
        print "Edge List: ", image.edgelist
        print
        print

    for image in queries:
        print "Name", image.file_name
        print image.decisionTree(database)

if __name__ == '__main__':
    main(sys.argv)
