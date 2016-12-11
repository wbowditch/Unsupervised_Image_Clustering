import sys
import os
from Image import *
import numpy as np
#from skimage import measure
#from countours import *
#from scipy import ndimage
#from scipy import misc
#from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage


"""  when you find the image and are trying ot scale it, you calculate the four corners and then find a buffer such that the array is square """
def main(argv):
    print argv
    database = []
    queries = []
    os.getcwd()

    for file in os.listdir("/Users/williambowditch/Documents/Algorithims/Unsupervised_Image_Clustering/database"):
        print file
        x = Image("database/"+file)
        database.append(x)

    for file in os.listdir("/Users/williambowditch/Documents/Algorithims/Unsupervised_Image_Clustering/queries"):
        print file
        x = Image("queries/"+file)
        queries.append(x)



    for image in queries:
        print "Name",image.file_name
        print image.s_z_r_b_matrix
        print "Area",image.area_
        print "b_Area",image.b_area_
        print "Center",image.b_center
        print "Angle",image.b_radians
        print "Corner Count", len(image.corners)
        print "Corners Group Count",len(image.grouped_corners)
        print "Corner Groups",image.grouped_corners
        print
        print
        print
        print


    for image in database:
        print "Name",image.file_name
        print image.s_z_r_b_matrix
        print "Area",image.area_
        print "b_Area",image.b_area_
        print "Center",image.b_center
        print "Angle",image.b_radians
        print "Corner Count", len(image.corners)
        print "Corners Group Count",len(image.grouped_corners)
        print "Corner Groups",image.grouped_corners
        print
        print
        print
        print

    for image in queries:
        print "Name",image.file_name
        print image.decisionTree(database)




        #print image.matrix
        #print image.calculate_ratios()
        #a = image.cornerDetector()
        #print image.createPockets()
        #print a
        #print image.intensityMap()
        #print image.buildPockets()
        #print a
        # for pair in a:
        #     image.matrix[pair[0]][pair[1]] = 5
        # print image.matrix
        # # #rint image.matrix
        # print
        # print





        # a = image.findCorners(6,0.05,20)
        #print a
        #print image.matrix
        # print a
        # for x,y,z in a:
        #     n = 5 if z<6 else 6
        #     image.matrix[x][y] = n
        # print image.matrix
        # print
        #print image.hamming_distance(first.matrix)
        #print abs(image.area() -a)
        #print image.matrix
        #print abs(image.calculate_ratios() -b)
        #print
        #print

        #print i
        #a = image.mean_average_blur()

        #np.savetxt("blur2/"+str(i)+".txt",a.astype(int),fmt="%d")
        #print

        # print
        # a = image.scale(2,2)
        # print a
        # np.savetxt("threes_scaled/"+str(i)+".txt",a.astype(int),fmt="%d")
        #i+=1
    #print image_array

    # index = 0
    # for image in image_array:
    #     file = open("threes_out/"+str(index)+".txt",'w')
    #
    #     # a = image.matrix
    #     # print a
    #     # #print
    #     # #flip_ud_face = np.flipud(a)
    #     # #print flip_ud_fa
    #     # print
    #     # #blurred = image.mean_average_blur(alpha = 2)
    #     rads = image.axis_of_least_second_movement()
    #     # #print blurred
    #     a = image.matrix
    #     rotate_face = ndimage.rotate(a, -math.degrees(rads),reshape=False)
    #     for line in rotate_face:
    #         x = ' '.join([str(x) for x in line])
    #         file.write(x + '\n')
    #     file.close()
    #     index+=1
    #     print index
    #     if index>20:
    #         break

        #print rotate_face
        # print
        # print
        # print
    #face = misc.face(gray=True)
    #a = np.array(image1)


    #blurred_face = ndimage.gaussian_filter(face, sigma=3)
   # print blurred_face
    #contours = find_contours(image1, 0.8)
    #print contours
    #print image1.center_of_area()
    #print image1.area()
    # a = image1.reshape()
    # print "original"
    # for x in image1.matrix:
    #     print x
    # print "cleaned"
    # for x in a:
    #     print x
    #queryPath = argv[1]
    #databasePath = argv[2]
    #k = argv[3]
    #os.listdir("somedirectory")

    #query_images = readImages(queryPath)
    #database_images = readImages(databasePath)
    #return output_array
    #return image_array

if __name__ == '__main__':
    main(sys.argv)

#def readImages(filePath):
