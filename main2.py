import os
import sys
from Image import Image
import time


def readImages(queryPath, databasePath, n):
    query_files = os.listdir(queryPath)
    database_files = os.listdir(databasePath)
    query_images = []
    database_images = []
    start = time.time()

    for i in range(n):
        file = query_files[i]
        #print queryPath + file
        img = Image(queryPath+"/"+file)
        query_images.append(img)
    end = time.time()
    count = 0
    start = time.time()
    for i in range(n):
        file = database_files[i]
        #print count
        #count += 1
        #print databasePath+"/"+file
        img = Image(databasePath+"/"+file)
        database_images.append(img)
    end = time.time()
    return query_images, database_images


def main(argv):
    i = 3000
    while (i < 6131):
        if (len(argv) >= 3):
            queryPath = argv[1]
            databasePath = argv[2]
            start = time.time()
            readImages(queryPath, databasePath,i)
            end = time.time()
            file = open("category_database/runtimes/runtime.txt", 'a')
            file.write("%d\t%f\n" % (i, end - start))
            file.close()
        i += 500
if __name__ == '__main__':
    main(sys.argv)


