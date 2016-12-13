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
<<<<<<< HEAD
        if not file.startswith('.'):
            #print file
            x = Image("database/"+file)
            database.append(x)
=======
        print file
        x = Image("database/"+file)
        database.append(x)
>>>>>>> 51d565dbe2df79ee4c7ea14c8b3d784ed2654bc5


    for file in os.listdir("queries"):
        if not file.startswith('.'):
            #print file
            x = Image("queries/"+file)
            queries.append(x)

    # for image in queries:
    #     print "Name", image.file_name
    #     print image.r_s_z_b_c_matrix
    #     print "Area", image.area_
    #     print "b_Area", image.b_area_
    #     print "Center", image.b_center
    #     print "Angle", image.b_radians
    #     print "Corner Count", len(image.corners)
    #     print "Corners Group Count", len(image.grouped_corners)
    #     print "Corner Groups", image.grouped_corners
    #     print
    #

    # for image in database:
    #     print "Name", image.file_name
    #     for shape in image.shapes:
    #         print "center"
    #         print shape.center
    #         print "max_r: {}\tmin_r: {}\tmax_c: {}\tmin_c: {}" .format(shape.max_r, shape.min_r, shape.max_c, shape.min_c)
    #         print "centered matrix"
    #         for line in shape.centered_matrix:
    #             print ' '.join(map(str, line))
    #         print "axis of least movement: {}".format(shape.theta)
    #         print "rotated matrix"
    #         for line in shape.rotated_matrix:
    #             print ' '.join(map(str, line))
    #         print "scaled matrix"
    #         for line in shape.scaled_matrix:
    #             print ' '.join(map(str, line))

    # Ryan testing
    # for image in database:
    #     print "File Name: " + image.file_name
    #     print "original:"
    #     for line in image.original_matrix:
    #         print ' '.join(map(str, line))
    #     print"final"
    #     for line in image.r_s_z_b_c_matrix:
    #         print ' '.join(map(str, line))

    #Decision tree - actual answer
    for image in queries:
        print "Name:\t{}".format(image.file_name)
        print
        for shape in image.shapes:
            for row in shape.clean_matrix:
                print ' '.join(map(str, row))
            print "Coords: ({},{},{},{})".format(shape.max_r,shape.min_r,shape.max_c,shape.min_c)
            print "height: {}\twidth: {}".format(shape.height_clean,shape.width_clean)
            print
            print "Centered"
            for row in shape.centered_matrix:
                print ' '.join(map(str,row))
            print
            print "rotate"
            for row in shape.rotated_matrix:
                print ' '.join(map(str, row))
            print
            print "Coords: ({},{},{},{})".format(shape.max_r_rotate, shape.min_r_rotate, shape.max_c_rotate, shape.min_c_rotate)
            print
            print "scaled"
            for row in shape.scaled_matrix:
                print ' '.join(map(str,row))
            print "Coords: ({},{},{},{})".format(shape.max_r_scale, shape.min_r_scale, shape.max_c_scale, shape.min_c_scale)
            print "height: {}\twidth: {}".format(shape.height_scale, shape.width_scale)
            #print image.compare(database)


if __name__ == '__main__':
    main(sys.argv)