from scipy.ndimage.filters import gaussian_filter
import sys
import os
from Image import *

queryPath = ""
def readImages():
    image_array = []
    os.getcwd()
    for file in os.listdir("/Users/David/Dropbox/2016F/CSCI3383/Unsupervised_Image_Clustering/database"):
        print file
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
    print "Started"

if __name__ == '__main__':
    main(sys.argv)
