from scipy.ndimage.filters import gaussian_filter
import sys
import os
from Image import *

queryPath = ""
def readImages():
    image_array = []
    os.getcwd()
    for file in os.listdir("/Users/David/Dropbox/2016F/CSCI3383/Unsupervised_Image_Clustering/database"):
        # print file
        x = Image(os.getcwd()+"/database/"+file)
        image_array.append(x)
    i = 0
    first = image_array[0]
    print first.matrix
    b =first.calculate_ratios()
    a = first.area()
    print
    for image in image_array[1:]:
        #print image.matrix
        #print image.calculate_ratios()
        a = image.cornerDetector()
        #print image.createPockets()
        #print a
        #print image.intensityMap()
        print image.buildPockets()
        #print a
        for pair in a:
            image.matrix[pair[0]][pair[1]] = 5
        print image.matrix
        # #rint image.matrix
    return image_array

def test():
    image_array = []
    os.getcwd()
    for file in os.listdir("/Users/David/Dropbox/2016F/CSCI3383/Unsupervised_Image_Clustering/database"):
        x = Image(os.getcwd() + "/database/" + file)
        image_array.append(x)
    image = image_array[6]
    edgelist = []
    edge_horizont = ndimage.sobel(image.s_z_r_b_matrix, 0)
    edge_vertical = ndimage.sobel(image.s_z_r_b_matrix, 1)
    magnitude = np.hypot(edge_horizont, edge_vertical)
    print image.s_z_r_b_matrix
    print edge_horizont
    print edge_vertical
    print magnitude
    for x in range(len(magnitude)):
        for y in range(len(magnitude[x])):
            if magnitude[x][y] != float(0) and magnitude[x][y] < float(1.5):
                edgelist.append((x, y))
    print edgelist
    tempimage = image.s_z_r_b_matrix
    for (x,y) in edgelist:
        tempimage[x][y] = 9
    print tempimage


def noisereduction():
    a = readImages()
    blurred = gaussian_filter(a, sigma=7)

def getInensityGradient(images):
    #filter with a Sobel kernel in  horizontal direction

    #filter with a Sobel kernel in  vertical direction
    G_x = 1
    H_x = 1
    edgeGradient = {}
    for pixel in images:
        edgeGradient = math.sqrt(G_x**2 + H_x**2)
        direction = math.atan(G_x/H_x)

# full scan of image is done to remove any unwanted pixels which may not constitute the edge
def nonMaximumSurpression():
    #at every pixel, pixel is checked if it is a local maximum in its neighborhood in the direction of gradient
    return 0
def main(argv):
    test()

if __name__ == '__main__':
    main(sys.argv)
