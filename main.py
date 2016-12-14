import sys
import os
from Image import *
from scipy.cluster.vq import whiten,vq,kmeans,kmeans2

"""  when you find the image and are trying ot scale it, you calculate the four corners and then find a buffer such that the array is square """


def main(argv):
    print argv
    database = []
    queries = []
    os.getcwd()
    dimensions = 0
    centroids_d = 0
    database_shapes = []
    query_shapes = []
    assignment_dictionary = {}
    query_dictionary = {}
    database_index = 0
    query_index = 0

    print "Opening Database..."
    for file in os.listdir("database"):
        if not file.startswith('.'):
            x = Image("database/"+file)
            if(x.object_count!= 1):
                continue
            dimensions+=x.object_count

            database_shapes.extend(x.shapes)
            for shape in x.shapes:
                assignment_dictionary[database_index] = x.file_name
                database_index+=1
            database.append(x)
    print "Opening queries..."
    for file in os.listdir("queries"):
        if not file.startswith('.'):
            if (x.object_count != 1):
                continue
            x = Image("queries/"+file)
            centroids_d += x.object_count
            query_shapes.extend(x.shapes)
            for shape in x.shapes:
                query_dictionary[query_index] = x.file_name
                query_index+=1
            queries.append(x)

    database_features = np.empty((dimensions,11),dtype='float64')
    query_features = np.empty((centroids_d,11),dtype='float64')
    for r in range(dimensions):
        database_features[r] = database_shapes[r].getFeatures()

    for r in range(centroids_d):
        query_features[r] = query_shapes[r].getFeatures()


    print database_features.shape
    print query_features.shape

    print "Generating Code Book"
    print query_dictionary[1]

    obs = whiten(database_features)
    k_guess = whiten(query_features)


    #print obs.shape

    #codebook,distortion = kmeans(obs,10)
    code, dist = vq(obs, k_guess)
    final_output = [[] for i in range(centroids_d)]

    for i in range(len(code)):
        final_output[code[i]].append( (assignment_dictionary[i],dist[i]) )

    for x in final_output:
        print x






    #print centroid.shape
    #print label.shape
    #for# i in range(len(label)):
        #print assignment_dictionary[i],i

    #print codebook,distortion
    #     #print distortion
    # d = {}
    # code, dist = vq(obs,codebook)
    # #print code
    # print database_index
    # for j in range(len(code)):
    #     i = code[j]
    #     print i
    #
    #     if i not in d:
    #         d[i] =[assignment_dictionary[j]]
    #     else:
    #         d[i].append(assignment_dictionary[j])
    #
    # for key in d:
    #     print d[key]
    #     print



    #print code,dist


    #Create a dictionary where the key is the M X N feature and the file
if __name__ == '__main__':
    main(sys.argv)