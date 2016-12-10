import sys

def main(argv):
    print argv
    output_array = []
    queryPath = argv[1]
    databasePath = argv[2]
    query_images = readImages(queryPath)
    database_images = readImages(databasePath)
    return output_array

if __name__ == '__main__':
    main(sys.argv)

#def readImages(filePath):
