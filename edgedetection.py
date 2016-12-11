import math
from scipy.ndimage.filters import gaussian_filter
import sys
import os
from Image import *
import numpy as np

queryPath = ""
def readImages():
    image_array = np.array([])
    os.getcwd()
    for file in os.listdir("/Users/David/Dropbox/BC/Fall16/Algorithms/Unsupervised_Image_Clustering"):
        x = Image(os.getcwd()+"/category_database/test_images/"+file)
        np.append(x)
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
    # at every pixel, pixel is checked if it is a local maximum in its neighborhood in the direction of gradient

def main(argv):
    print readImages()

if __name__ == '__main__':
    main(sys.argv)