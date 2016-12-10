import sys
import os
from Image import *
def main(argv):
    print argv
    image_array = []
    os.getcwd()

    for file in os.listdir("/Users/williambowditch/Documents/Algorithims/Unsupervised_Image_Clustering/category_database/test_images"):
        x = Image(os.getcwd()+"/category_database/test_images/"+file)
        image_array.append(x)
    print image_array
    #queryPath = argv[1]
    #databasePath = argv[2]
    #k = argv[3]
    #os.listdir("somedirectory")

    #query_images = readImages(queryPath)
    #database_images = readImages(databasePath)
    #return output_array
    return image_array

if __name__ == '__main__':
    main(sys.argv)

#def readImages(filePath):
