#All text files will be read into this class
import math
import numpy as np
from scipy import ndimage
import operator
from numpy.linalg import svd
class Image(object):

    def __init__(self,file_name,rows=0,cols=0):
        self.file_name = file_name.split('/')[-1]
        self.original_matrix = self._create_matrix(file_name)
        self.area_ = self.area()
        self.rows = self.original_matrix.shape[0]
        self.cols = self.original_matrix.shape[1]
        self.size = len(self.original_matrix)*len(self.original_matrix[0])
        # print "ORIGINAL"
        # print self.original_matrix
        # print

        #BLUR Image, get area and coordinates of blurred image object
        self.b_matrix = self.mean_average_blur()
        self.b_area_ = self.b_area()
        self.b_center = self.b_center_of_area()
        #
        # print "BLURRED"
        # print self.b_matrix
        # print

        self.b_radians = self.b_axis_of_least_second_movement()
        self.r_b_matrix = self.rotate_blurred_matrix()


        # print "ROTATED"
        # print self.r_b_matrix
        # print

        self.z_r_b_matrix = self.zoom()








        # print "ZOOMED"
        # print self.z_r_b_matrix
        # print

        self.z_rows = self.z_r_b_matrix.shape[0]
        self.z_cols = self.z_r_b_matrix.shape[1]

        self.scale_cols = self.cols/self.z_cols
        self.scale_rows = self.rows/self.z_rows

        self.s_z_r_b_matrix = self.scale()
        self.corners = self.cornerDetector()

        self.grouped_corners = self.buildPockets()

        # img = self.s_z_r_b_matrix
        # for x,y in self.grouped_corners:
        #     img[x][y] = 5
        # print img

        self.neighborhoods = self.corner_neighborhood()

        self.edges = self.getEdgeList()

        self.edge_groups = self.edges_neighborhood()

        # self.pdist = self._pdist()
        # print self.pdist

        # print self.size
        #
        # print len(self.s_z_r_b_matrix)*len(self.s_z_r_b_matrix[0])









        #DEBATABLE WHETHER WE NEED THESE POINTS
        # self.north = self.north()
        # self.south = self.south()
        # self.east = self.east()
        # self.west = self.west()
        #print self.matrix
        #self.matrix = self.mean_average_blur()


    def _create_matrix(self,name): #reads image from textfile into numpy array
        file = open(name,'r')
        array = []
        for line in file:
            #print line
            array.append([int(x) for x in line.split(" ")])
        return np.array(array)

    def rotate_blurred_matrix(self):
        rads = self.b_radians
        return ndimage.rotate(self.b_matrix, -math.degrees(rads),reshape=False)


    def decisionTree(self, database_images,k=9,
                     area_sigma=30,
                     b_area_sigma=30,
                     center_diff_sigma=5,
                     rads_sigma=0.01,
                     scale_cols_sigma=1,
                     scale_rows_sigma=1,
                     hamming_simga1=.8,
                     hamming_simga2=.85,
                     hamming_simga3=.9,
                     hamming_simga4=.9,
                     ):  #return k closets neighbors
        euclideanDistance = lambda a,b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        score =  {image: 0 for image in database_images}

        for image in database_images:
            points = 0
            if abs(self.area_ - image.area_) < area_sigma:
                points+=2
            else:
                points-=1

            if abs(self.b_area_ - image.b_area_) < b_area_sigma:
                points+=2
            else:
                points-=1

            if euclideanDistance(self.b_center,image.b_center) < center_diff_sigma:
                points+=2
            else:
                points-=1

            if abs(self.b_radians - image.b_radians) == rads_sigma:
                points+=2
            else:
                points-=1

            if abs(self.scale_cols - image.scale_cols) < scale_cols_sigma:
                points+=2
            else:
                points-=1

            if abs(self.scale_rows - image.scale_rows) < scale_rows_sigma:
                points+=2
            else:
                points-=1

            if self.hamming_distance4(image.s_z_r_b_matrix) > hamming_simga1:
                points+=2
            else:
                points-=1

            if self.hamming_distance4(image.s_z_r_b_matrix) > hamming_simga2:
                points+=2
            else:
                points-=1

            if self.hamming_distance4(image.s_z_r_b_matrix) > hamming_simga3:
                points+=3

            if self.hamming_distance4(image.s_z_r_b_matrix) > hamming_simga4:
                points+=4


            #points+= - abs(len(self.neighborhoods)-len(image.neighborhoods))**2

            c = 0
            #a = self.edges
            for neighborhood1 in self.neighborhoods:
                #print svd(neighborhood1)
                U1= svd(neighborhood1,compute_uv=False)
                for neighborhood2 in image.neighborhoods:

                    U2= svd(neighborhood2,compute_uv=False)

                    #print svd(neighborhood1) - svd(neighborhood2,compute_uv=False)
                    diff = abs(sum(U1 - U2))

                    # print diff
                    if diff==0:
                        # print self.file_name,image.file_name,round(diff,2)
                        # print neighborhood1
                        # print neighborhood2
                        # print self.s_z_r_b_matrix
                        # print image.s_z_r_b_matrix
                        points+=2
                        c+=1




                    #else:
                        #print self.file_name,image.file_name,round(diff,2)

            #     x,y = corner
            #     block = neighbors = [(a,b) for a in range(x-2,x+3) for b in range(y-2,y+3)]
            #     block.remove((x,y))
            #     for point in block:
            #         if point in image.grouped_corners:
            #             points+=1
            # #print points
            # print image.file_name,points
            score[image] = points

        out = []
        d = dict(sorted(score.iteritems(), key=operator.itemgetter(1), reverse=True)[:int(k)])
        count = 0
        for key in d:
            out.append(key.file_name)
            if key.file_name.startswith(self.file_name[:3]):
                count+=1
        return out,count




    def corner_neighborhood(self):
        corners = self.grouped_corners
        img = self.s_z_r_b_matrix
        neighborhoods = []

        for r,c in corners:
            try:
                neighborhood = np.array([img[a][b] for a in range(r-2,r+3) for b in range(c-2,c+3)]).reshape((5,5))
            except IndexError:
                continue
            #print neighborhood
            neighborhoods.append(neighborhood)

        return neighborhoods


    def edges_neighborhood(self):
        corners = self.edges
        img = self.s_z_r_b_matrix
        neighborhoods = []
        for r,c in corners:
            try:
                neighborhood = np.array([img[a][b] for a in range(r-2,r+3) for b in range(c-2,c+3)]).reshape((5,5))
            except IndexError:
                continue
            #print neighborhood
            neighborhoods.append(neighborhood)

        return neighborhoods



    def zoom(self):
        r,c = self.b_center
        img = self.r_b_matrix
        #print r,c

        side_rows = round((self.b_area_)**.5)
        side_cols = side_rows
        #print "before"
        #print side_cols,side_rows
        #print self.rows,self.cols

        return img

        while self.rows%side_rows !=0: #so that it scaled properly
            side_rows+=1
        while self.cols%side_cols !=0: #so that it scaled properly
            side_cols+=1

        print side_rows,side_cols

        offset_r = side_rows
        offset_c = side_cols+1
        x = img[max(0,r-offset_r):min(r+offset_r,self.rows),max(0,c-offset_c):min(self.cols,c+offset_c)]
        return x





    def mean_average_blur(self,alpha = 10): #blures the object, removes noise
        image_array = self.original_matrix
        image_output = image_array.copy()
        height=len(image_array)
        width = len(image_array[0])
        image_array[0] = [0]*width
        image_array[height-1] = [0]*width
        for x in range(height):
            image_array[x][0]=0
            image_array[x][width-1]=0

        for row in range(height):
            for col in range(width):
                #rowStart = row/alpha*alpha
                #colStart = col/alpha*alpha
                neighbors = [image_array[x][y] for x in range(max(row-2,0),min(row+2,height)) for y in range(max(0,col-2),min(col+2,width))]
                ones = neighbors.count(1)
                zeros = neighbors.count(0)
                image_output[row][col] = 1 if ones>=zeros else 0
                #neighbors.countsum([p for p in pixelList])/len(pixelList)
            #image_array[row][col][1] = sum([p[1] for p in pixelList ])/len(pixelList)
                #image_array[row][col][2] = sum([p[2] for p in pixelList ])/len(pixelList)
        #self.matrix = image_output
        return image_array
        #return image_output

    def __len__(self):
        return self.size

    def rows(self):
        return self.rows

    def cols(self):
        return self.cols

    def b_area(self):
        return self.b_matrix.sum()

    def area(self):
        return self.original_matrix.sum()



    def b_center_of_area(self): #returns estimated center of object
        img = self.b_matrix
        #self.area_ = self.b_area
        r_=0
        c_=0
        for r in range(self.rows):
            for c in range(self.cols):
                r_+= r*img[r][c]
                c_+= c*img[r][c]
        a = 1./self.b_area_
        r_ = r_*a
        c_ = c_*a

        return int(round(r_)),int(round(c_))



    def b_axis_of_least_second_movement(self): #returns the radian degree of rotation
        a = 0.0
        b = 0.0
        c = 0.0
        img = self.b_matrix
        c_,r_ = self.b_center_of_area()
        for r in range(self.rows):
            for c in range(self.cols):
                a+= (r - r_)*(c-c_)*img[r][c]
                b+= (r-r_)**2*img[r][c]
                c+= (c-c_)**2*img[r][c]
        out = 2.*a/(b - c)
        return 0.5*math.atan(out)


    def hamming_distance1(self,arr2):
        img = self.original_matrix
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared+=1
        return shared/self.size


    def hamming_distance2(self,arr2):
        img = self.b_matrix
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared+=1
        return shared/self.size


    def hamming_distance3(self,arr2):
        img = self.r_b_matrix
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared+=1
        return shared/self.size


    def hamming_distance4(self,arr2):
        img = self.s_z_r_b_matrix
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared+=1
        return shared/self.size




    def scale(self): #Kronecker product
        #scale_x = self.rows/sub_rows
        #scale_y = self.cols/sub_cols
        #scale_x = self.scale_cols
        #scale_y = self.scale_

        a = np.kron(self.z_r_b_matrix, np.ones((self.scale_rows,self.scale_cols)))
        # alternate = True
        # while a.shape[0] > self.rows:
        #     if alternate:
        #         a = a[1:]
        #     else:
        #         a = a[:a.shape[1]-1]
        #
        # while a.shape[1] > self.cols:
        #     if alternate:
        #         a = a[:][1:]
        #     else:
        #         a = a[:][:self.a.shape[1]-1]

        return a.astype(int)



    def north(self):
        for i in range (self.rows):
            for j in range (self.cols):
                if self.matrix[i][j] == 1:
                    return i

    def south(self):
        for i in range (self.rows-1, 0, -1):
            for j in range(self.cols-1, 0, -1):
                if self.matrix[i][j] == 1:
                    return i

    def east(self):
        for i in range (self.cols-1, 0, -1):
            for j in range (self.rows-1, 0, -1):
                if self.matrix[j][i] == 1:
                    return j

    def west(self):
        for i in range (self.cols):
            for j in range (self.rows):
                if self.matrix[j][i] == 1:
                    return j


    def calculate_ratios(self):
        #print self.north,self.south,self.east,self.west
        # north_south = np.linalg.norm(self.north-self.south)
        # north_west = np.linalg.norm(self.north-self.west)
        # east_west = np.linalg.norm(self.west - self.east)
        # north_east = np.linalg.norm(self.north-self.east)
        # south_west = np.linalg.norm(self.south - self.west)
        # south_east = np.linalg.norm(s
        # elf.south - self.east)
        #print north_south,east_west
        dy, dx = np.gradient(self.matrix)
        north_south = self.south-self.north
        east_west = self.east - self.west

        #north_south = self.south-self.north
        #east_west = self.east - self.west

        ns_ew = float(north_south)/float(east_west)

        #nw_se = north_west/south_east

        #ne_sw = north_east/south_west
        return dy,dx
        #return ns_ew#,nw_se,ne_sw





    def cornerDetector(self):
        corners = []
        image_array = self.s_z_r_b_matrix
        rows = self.rows
        cols = self.cols

        for i in range(rows):
            for j in range(cols):
                if image_array[i][j] ==1:
                    neighbors = [image_array[x][y] for x in range(max(i-1,0),min(i+2,rows)) for y in range(max(0,j-1),min(j+2,cols))]
                    #print neighbors
                    if neighbors.count(0)>4:
                        corners.append((i,j))
        self.corners = corners
        return corners


    def buildPockets_recurse(self,t,corners):
        x,y = t[0],t[1]
        #print "looking at neighbors of",(x,y),
        #print "corners is currently",corners
        if len(corners) == 0:
            return []
        pocket = []
        neighbors = [(a,b) for a in range(x-2,x+3) for b in range(y-2,y+3)] # grab the nearest corners
        neighbors.remove((x,y))
        for z,w in neighbors:
            if (z,w) in corners:

                corners.remove((z,w))
                pocket.extend([(z,w)] + self.buildPockets_recurse((z,w),corners))
        return pocket


    def buildPockets(self):
        pockets = []
        corners = list(self.corners)
        getKey = lambda a : math.sqrt(a[0]**2+a[1]**2)
        corners = sorted(corners,key=getKey)
        #print corners
        for i in range(len(self.corners)):
            if not corners:
                break
            x,y = corners.pop(0) # pop off the first corner
            pocket = [(x,y)]
            neighbors = [(a,b) for a in range(x-2,x+3) for b in range(y-2,y+3)] # grab the nearest corners
            neighbors.remove((x,y))
            for z,w in neighbors:
                if (z,w) in corners:
                    #print "found neighbor",(z,w)
                    corners.remove((z,w))

                    pocket.extend([(z,w)] + self.buildPockets_recurse((z,w),corners))

            pockets.append(pocket)
            #print "pockets",pockets
        pockets_averaged = []
        #print "here",pockets
        for group in pockets:
            #print group
            avg_x = sum([p[0] for p in group])/len(group)
            avg_y = sum([p[1] for p in group])/len(group)
            pockets_averaged.append((avg_x,avg_y))
        #print 1
        return pockets_averaged





    #
    # def cornerDetectorv3(self,alpha):
    #     corners = []
    #     image_array = self.matrix
    #     rows = self.rows
    #     cols = self.cols
    #     squares = alpha**2
    #     for row in range(0,rows,alpha):
    #         for col in range(0,cols,alpha):
    #             neighbors = [image_array[x][y] for x in range(row,min(row+alpha,rows)) for y in range(col,min(col+alpha,cols))]
    #             ones = neighbors.count(1)
    #             zeros = neighbors.count(0)
    #             if zeros and ones > alpha*2:
    #                 corners.append((row+alpha/2,col+alpha/2))
    #
    #     return corners
    #             #image_output[row][col] = 1 if ones>=zeros else 0
    #             #neighbors.countsum([p for p in pixelList])/len(pixelList)
    #         #image_array[row][col][1] = sum([p[1] for p in pixelList ])/len(pixelList)
    #             #image_array[row][col][2] = sum([p[2] for p in pixelList ])/len(pixelList)
    #     #self.matrix = image_output
    #     #return image_output







        #diff = [corners[i+1]-corners[i] for i in range(len(corners)-1)]

        #for x,y in self.corners:



    # def getNeighbors(self,corners, k):
    #     euclideanDistance = lambda a,b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    #     potential_pockets = []
    #     for x in corners:
    #         pockets = []
    #         for y in corners:
    #             if euclideanDistance(x,y)<
    #
    #
    #     distances = []
    #     length = len(testInstance)-1
    #     for x in range(len(trainingSet)):
    #         dist = self.euclideanDistance(testInstance, trainingSet[x], length)
    #         distances.append((trainingSet[x], dist))
    #     distances.sort(key=operator.itemgetter(1))
    #     neighbors = []
    #     for x in range(k):
    #         neighbors.append(distances[x][0])
    #     return neighbors









    def intensityMap(self):
        #corners = []
        image_array = self.matrix
        image_out = image_array
        rows = self.rows
        cols = self.cols
        for i in range(rows):
            for j in range(cols):
                if image_array[i][j] ==1:
                    neighbors = [image_array[x][y] for x in range(max(i-1,0),min(i+2,rows)) for y in range(max(0,j-1),min(j+2,cols))]
                    #print neighbors
                    image_out[i][j] = neighbors.count(1)

        return image_out

    def distances(self):
        points = self.grouped_corners
        if(len(points) > 1):
            distances = {}
            for i in range(len(points)):
                a = points[i]
                for j in range(i + 1, len(points)):
                    b = points[j]
                    distances[(a,b)] = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            maximum = max(distances, key=distances.get)
            print "Max distance between " + str(maximum) + "= " + str(distances[maximum])
            minimum = min(distances, key=distances.get)
            print "Min distance between " + str(minimum) + "= " + str(distances[minimum])
        else:
            print "Only " + str(len(points)) + " point(s)"


    def getEdgeList(self):
        edgelist = []
        edge_horizont = ndimage.sobel(self.s_z_r_b_matrix, 0)
        edge_vertical = ndimage.sobel(self.s_z_r_b_matrix, 1)
        magnitude = np.hypot(edge_horizont, edge_vertical)
        for x in range(len(magnitude)):
            for y in range(len(magnitude[x])):
                if magnitude[x][y] != float(0):
                    edgelist.append((x, y))
        return edgelist


















