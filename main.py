import sys
import os
from Image import *

"""  when you find the image and are trying ot scale it, you calculate the four corners and then find a buffer such that the array is square """


def main(argv):
    print argv
    database = []
    queries = []
    os.getcwd()

    for file in os.listdir("compare/bones"):
        if not file.startswith('.'):
            print file
            x = Image("compare/bones/"+file)
            #print x.shapes[0].shape_corners
            print "initialize",file
            database.append(x)


    # for file in os.listdir("queries"):
    #     if not file.startswith('.'):
    #         print file
    #         x = Image("queries/"+file)
    #         queries.append(x)

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
    for image in database:
        shape = image.shapes[0]
        #corners2 = shape.shape_corners
        print image.file_name
        #print image.original_matrix
        print shape.clean_matrix.sum()
        print shape.scaled_matrix.sum()
        print "size",shape.size_scale
        print "area", shape.area_scale

        print "area/size",shape.area_to_size
        print "height/width",shape.height_to_width_ratio() #height / width
        file = open("compare/outputs/"+image.file_name,'w')
        for row in shape.zoomed_matrix:
            line1 = [str(int(x)) for x in row]
            line = ' '.join(line1)
            file.write(line + '\n')
        file.close()
        #print shape.scaled_matrix
        #for rc, in shape.scaled_matrix

        #corners = shape.shape_grouped_corners
        #print shape.scaled_matrix
       # matrix = shape.scaled_matrix.copy()
        #for r,c in corners:
         #   matrix[r][c] = 5
        #print matrix

        #print shape.max_r_rotate, shape.min_r_rotate, shape.max_c_rotate, shape.min_c_rotate
        #print shape.height_scale
        #print shape.width_scale
        #print shape.height_to_width_ratio()

    #print database[0].compare(database)
    # for image in queries:
    #     print "Name",image.file_name
    #     print image.original_matrix
    #     print image.compare(database)


if __name__ == '__main__':
    main(sys.argv)