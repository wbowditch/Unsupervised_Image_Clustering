#All text files will be read into this class
import math
import numpy as np
from scipy import ndimage
import operator
from numpy.linalg import svd
class Image(object):

    def __init__(self, file_name, rows=0, cols=0):
        self.file_name = file_name.split('/')[-1]
        self.original_matrix = self._create_matrix(file_name)
        self.area_ = self.area()
        self.shapes = []
        if self.area_== 0:
            self.empty = True

        else:

            self.empty = False  #yay it's not blank

            self.invert = self.invert_or_not()
            if(self.invert):
                self.original_matrix = self.invert_matrix()
                self.area_ = self.area()

            self.rows,self.cols = self.original_matrix.shape
            self.size = self.rows*self.cols

            self.objects = self.findObjects()
            if len(self.objects) !=1:
                self.cleaned_objects = self.cleanObjects()
            else:
                self.cleaned_objects = self.objects

            self.shapes = [Shape((self.rows,self.cols),obj) for obj in self.cleaned_objects]
            #print "HALLLO",len(self.shapes)





                #Only one object bruh


        # # BLUR Image, get area and coordinates of blurred image object
        # self.b_matrix = self.mean_average_blur()
        # self.b_area_ = self.b_area()
        # self.b_center = self.b_center_of_area()
        # self.b_radians = self.b_axis_of_least_second_movement()
        # self.b_c_matrix = self.b_center_matrix()

        # # Points
        # self.east_point = self.east()
        # self.west_point = self.west()
        # self.north_point = self.north()
        # self.south_point = self.south()

        # # Zoom
        # self.keep = self.keep_or_not()
        # self.z_b_c_matrix = self.zoom()  # old = self.z_r_b_matrix
        # self.z_rows = self.z_b_c_matrix.shape[0]
        # self.z_cols = self.z_b_c_matrix.shape[1]
        # self.scale_cols = self.cols / self.z_cols
        # self.scale_rows = self.rows / self.z_rows

        # # Scale
        # self.s_z_b_c_matrix = self.scale()
        # self.r_s_z_b_c_matrix = self.rotate_blurred_matrix()  # old = self.s_z_r_b_matrix
        # self.final_area = self.area()

        # Extra Features
        # self.corners = self.cornerDetector()
        # self.grouped_corners = self.buildPockets()
        # self.neighborhoods = self.corner_neighborhood()
        # self.edges = self.getEdgeList()
        # self.edge_groups = self.edges_neighborhood()

        #Check Number of Objects
        #print len(self.objects)


    def _create_matrix(self,name):  # Reads image from textfile into numpy array
        file = open(name,'r')
        array = []
        for line in file:
            array.append([int(x) for x in line.split(" ")])
        return np.array(array)

    def rotate_blurred_matrix(self):
        rads = self.b_radians
        return ndimage.rotate(self.s_z_b_c_matrix, -math.degrees(rads),reshape=False)


    def invert_or_not(self):
        outer_sum = 0
        outer_sum += self.original_matrix[0].sum()
        outer_sum += self.original_matrix[-1].sum()
        outer_sum += self.original_matrix[:, 0].sum()
        outer_sum += self.original_matrix[:, -1].sum()
        size = (self.original_matrix.shape[0] + self.original_matrix.shape[1]) * 2
        if outer_sum >= int(size)/2.:
            return True
        else:
            return False

    def invert_matrix(self):
        if self.has_inverted_matrix:
            inverted_matrix = []
            for i in range(self.rows):
                inverted_matrix.append(1-self.original_matrix[i])
            return np.array(inverted_matrix)

    # def aggregate_comparison_vals(self, database_images):
    #     for image in database_images:

    def compare(self, database_images):  # return k closets neighbors
        euclideanDistance = lambda a, b: math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        thetas, image_areas, h_w_diffs, s_a_diffs, clean_hamming_dists, post_hamming_dists = [], [], [], [], [], []
        for image in database_images:
            print image.original_matrix
            if (self.empty):
                if (image.empty):  # only select other blank images
                    print "image", image.file_name, "is empty"
                    continue

            print "Image Area", abs(self.area_ - image.area_)

            print "Shape Count Difference", abs(len(self.shapes) - len(image.shapes))

            #print "hollow objects Difference", abs(self.hollowObject() - image.hollowObject())

            for query_shape in self.shapes:
                for database_shape in database_images.shapes:
                    #print "Database Img: {}" .format(database_images.)


                    for row in query_shape:
                        print ' '.join(map(str,row))
                    for row in database_shape:
                        print ' '.join(map(str,row))
                    print "Theta Difference: {}".format( abs(query_shape.theta - database_shape.theta))
                    thetas.append(abs(query_shape.theta - database_shape.theta))
                    print "Area Shape Difference: {}" .format(abs(query_shape.area_clean - database_shape.area_clean))
                    image_areas.append(abs(query_shape.area_clean - database_shape.area_clean))
                    print "height to width diff: {}".format( abs(query_shape.height_to_width_ratio() - database_shape.height_to_width_ratio()))
                    h_w_diffs.append(abs(query_shape.height_to_width_ratio() - database_shape.height_to_width_ratio()))
                    print "size to area diff: {}" .format(abs(query_shape.size_to_area_ratio() - database_shape.size_to_area_ratio()))
                    s_a_diffs.append(abs(query_shape.size_to_area_ratio() - database_shape.size_to_area_ratio()))
                    print "hamming clean", abs(query_shape.hamming_distance(query_shape.clean_matrix, database_shape.clean_matrix))
                    clean_hamming_dists.append(abs(query_shape.hamming_distance(query_shape.clean_matrix, database_shape.clean_matrix)))
                    print "hamming scaled", abs(query_shape.hamming_distance(query_shape.scaled_matrix, database_shape.scaled_matrix))
                    post_hamming_dists.append(abs(query_shape.hamming_distance(query_shape.scaled_matrix, database_shape.scaled_matrix)))
                    neighborhoods1 = query_shape.corner_neighborhood
                    neighborhoods2 = database_shape.corner_neighborhood
                    count = 0
                    for neighborhood1 in neighborhoods1:
                        U1 = svd(neighborhood1, compute_uv=False)
                        for neighborhood2 in neighborhoods2:

                            U2 = svd(neighborhood2, compute_uv=False)
                            diff = abs(sum(U1 - U2))
                            print neighborhood1
                            print neighborhood2
                            print diff
                            print
                            print
                            if diff == 0:
                                count += 1
                    print "Neighbors in common:", count

                    corners1 = query_shape.shape_grouped_corners
                    corners2 = database_shape.shape_grouped_corners
                    count = 0
                    corner_min = 100000000
                    for v1 in corners1:
                        for v2 in corners2:
                            if abs(euclideanDistance(v1, v2)) < corner_min:
                                corner_min = abs(euclideanDistance(v1, v2))
                    print "min corner", corner_min
                    # print "Corners:",count


    def decisionTree(self, database_images,k=9,
                     area_sigma=30,
                     center_diff_sigma=5,
                     rads_sigma=0.01,
                     scale_cols_sigma=1,
                     scale_rows_sigma=1,
                     shape_count = 0,
                     h_w_ratio=0.1,
                     s_a_ratio=0.1,
                     corners_sigma = 10


                     ):  # return k closets neighbors
        euclideanDistance = lambda a,b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        score = {image: 0 for image in database_images}

        for image in database_images:
            print "query image: {}\tdatabase image: {}" .format( self.file_name, image.file_name )
            points = 0
            if(self.empty):
                if(image.empty):  #only select other blank images
                    points+=1000
                score[image] = points
                continue

            if abs(self.area_ - image.area_) < area_sigma:
                points += 2
            else:
                points -= 1
            # print dir(image)
            # print image.shapes
            # print image.file_name
            if not image.shapes:
                print"no shapes"
                print image.file_name
                print image.area_
                for row in image.original_matrix:
                    print row
                print
                print
            if abs(len(self.shapes) - len(image.shapes)) < shape_count:
                points +=2
            else:
                points -= 1

            for query_shape in self.shapes:
                for database_shape in image.shapes:

                    print "query shape theta: {}\tdatabase image theta: {}" .format(query_shape.theta, database_shape.theta)

                    if abs(query_shape.theta - database_shape.theta) < rads_sigma:
                        points += 2

                    print "query h/w ratio: {}\tdatabase h/w ratio: {}" .format(query_shape.height_to_width_ratio(), database_shape.height_to_width_ratio())
                    if abs(query_shape.height_to_width_ratio() - database_shape.height_to_width_ratio()) < h_w_ratio:
                        points += 2
                    else:
                        points -= 1

                    print "query s/a ratio: {}\tdatabase s/a ratio: {}" .format(query_shape.size_to_area_ratio(), database_shape.size_to_area_ratio())
                    if abs(query_shape.size_to_area_ratio() - database_shape.size_to_area_ratio()) < s_a_ratio:
                        points += 2
                    else:
                        points -= 1

                    print "query pre-scale hamming dist: {}".format(query_shape.hamming_distance_prescale(database_shape.clean_matrix))
                    if abs(query_shape.hamming_distance_prescale(database_shape.clean_matrix)) >0.8:
                        points += 2
                    else:
                        points -= 1

                    print "post scale hamming dist: {}".format(query_shape.hamming_distance_prescale(database_shape.scaled_matrix))
                    if abs(query_shape.hamming_distance_postscale(database_shape.scaled_matrix)) >0.8:
                        points += 2
                    else:
                        points -= 1

                    neighborhoods1 = query_shape.corner_neighborhood
                    neighborhoods2 = database_shape.corner_neighborhood

                    for neighborhood1 in neighborhoods1:
                        U1 = svd(neighborhood1,compute_uv=False)
                        for neighborhood2 in neighborhoods2:
                            U2 = svd(neighborhood2,compute_uv=False)
                            diff = abs(sum(U1 - U2))
                            if diff == 0:
                                points += 2

                    corners1 = query_shape.shape_grouped_corners
                    corners2 = database_shape.shape_grouped_corners
                    for v1 in corners1:
                        for v2 in corners2:
                            if abs(euclideanDistance(v1,v2))< corners_sigma:
                                points+=2
            score[image] = points
        out = []
        d = dict(sorted(score.iteritems(), key=operator.itemgetter(1), reverse=True)[:int(k)])
        count = 0
        for key in d:
            out.append(key.file_name)
            if key.file_name.startswith(self.file_name[:3]):
                count += 1
        return out, count

    def corner_neighborhood(self):
        corners = self.grouped_corners
        img = self.r_s_z_b_c_matrix
        neighborhoods = []
        for r,c in corners:
            try:
                neighborhood = np.array([img[a][b] for a in range(r-2,r+3) for b in range(c-2,c+3)]).reshape((5,5))
            except IndexError:
                continue
            neighborhoods.append(neighborhood)
        return neighborhoods

    def edges_neighborhood(self):
        corners = self.edges
        img = self.r_s_z_b_c_matrix
        neighborhoods = []
        for r,c in corners:
            try:
                neighborhood = np.array([img[a][b] for a in range(r-2,r+3) for b in range(c-2,c+3)]).reshape((5,5))
            except IndexError:
                continue
            neighborhoods.append(neighborhood)
        return neighborhoods

    def zoom(self):
        img = self.b_c_matrix
        area_of_shape = self.b_area()
        north = self.north()
        west = self.west()
        south = self.south()
        east = self.east()
        x_distance , y_distance = 0, 0
        if self.keep == False:
            return self.b_c_matrix
        while self.rows % (south - north + y_distance) != 0:
            y_distance += 1
        while self.cols % (east - west + x_distance) != 0:
            x_distance += 1
        if ((south-north + y_distance) > self.rows-4/2) or ((east - west + x_distance) > self.cols-4/2):
            return self.b_c_matrix
        if y_distance % 2 == 1:
            south += 1
        if x_distance % 2 == 1:
            west -= 1
        north -= y_distance/2
        south += y_distance/2
        west -= x_distance/2
        east += x_distance/2
        if east > self.cols:
            while east != self.cols:
                east -= 1
                west -= 1
        if west < 0:
            while west != 0:
                west += 1
                east += 1
        if north < 0:
            while north != 0:
                north += 1
                south += 1
        if south > self.rows:
            while south != self.rows:
                north -= 1
                south -= 1
        temp = img[north:south, west:east]
        return temp

    def mean_average_blur(self, alpha = 10):  # blurs the object, removes noise
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
                neighbors = [image_array[x][y] for x in range(max(row-1, 0), min(row+1, height)) for y in range(max(0, col - 1), min(col + 1, width))]
                ones = neighbors.count(1)
                zeros = neighbors.count(0)
                image_output[row][col] = 1 if ones >= zeros and row != self.rows-1 and col != self.cols-1 else 0
        return image_output

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

    def f_area(self):
        return self.r_s_z_b_c_matrix.sum()

    def b_center_of_area(self):  # Returns estimated center of object
        img = self.b_matrix
        r_ = 0
        c_ = 0
        for r in range(self.rows):
            for c in range(self.cols):
                r_ += r*img[r][c]
                c_ += c*img[r][c]
        try:
            a = 1./self.b_area_
            r_ *= a
            c_ *= a
            return int(round(r_)), int(round(c_))
        except ZeroDivisionError:
            return 0, 0

    def b_axis_of_least_second_movement(self):  # Returns the radian degree of rotation
        a = 0.0
        b = 0.0
        c = 0.0
        img = self.b_matrix
        c_, r_ = self.b_center_of_area()
        for r in range(self.rows):
            for c in range(self.cols):
                a += (r - r_)*(c-c_)*img[r][c]
                b += (r-r_)**2*img[r][c]
                c += (c-c_)**2*img[r][c]
        out = 2. * a/(b - c)
        return 0.5 * math.atan(out)

    def hamming_distance1(self, arr2):
        img = self.original_matrix
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared += 1
        return shared/self.size

    def hamming_distance2(self, arr2):
        img = self.b_matrix
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared += 1
        return shared/self.size

    def hamming_distance3(self, arr2):
        img = self.b_c_matrix
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared += 1
        return shared/self.size

    def hamming_distance4(self,arr2):
        img = self.r_s_z_b_c_matrix
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared+=1
        return shared/self.size




    def scale(self): #Kronecker product
        a = np.kron(self.z_b_c_matrix, np.ones((self.scale_rows,self.scale_cols)))
        return a.astype(int)

    def north(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.b_c_matrix[i][j] == 1:
                    return i-2

    def south(self):
        for i in range(self.rows-1, 0, -1):
            for j in range(self.cols-1, 0, -1):
                if self.b_c_matrix[i][j] == 1:
                    return i+4

    def east(self):
        for i in range(self.cols-1, 0, -1):
            for j in range(self.rows-1, 0, -1):
                if self.b_c_matrix[j][i] == 1:
                    return i+4

    def west(self):
        for i in range(self.cols):
            for j in range(self.rows):
                if self.b_c_matrix[j][i] == 1:
                    return i-2

    def keep_or_not(self):
        if self.north_point < 0 or self.south_point > self.rows or self.west_point < 0 or self.east_point > self.cols or self.south_point-self.north_point > self.rows or self.east_point-self.west_point:
            return False
        else:
            return True

    def calculate_ratios(self):
        dy, dx = np.gradient(self.original_matrix)
        north_south = self.south - self.north
        east_west = self.east - self.west
        ns_ew = float(north_south)/float(east_west)
        return dy,dx

    # def cornerDetector(self):
    #     corners = []
    #     image_array = self.r_s_z_b_c_matrix
    #     rows = self.rows
    #     cols = self.cols
    #     for i in range(rows):
    #         for j in range(cols):
    #             if image_array[i][j] == 1:
    #                 neighbors = [image_array[x][y] for x in range(max(i-1, 0), min(i+2, rows)) for y in range(max(0, j-1), min(j+2, cols))]
    #                 if neighbors.count(0) > 4:
    #                     corners.append((i, j))
    #     self.corners = corners
    #     return corners

    def buildPockets_recurse(self, t, corners):
        x,y = t[0], t[1]
        if len(corners) == 0:
            return []
        pocket = []
        neighbors = [(a, b) for a in range(x-2, x+3) for b in range(y-2, y+3)]  # Grab the nearest corners
        neighbors.remove((x, y))
        for z, w in neighbors:
            if (z, w) in corners:

                corners.remove((z, w))
                pocket.extend([(z, w)] + self.buildPockets_recurse((z, w), corners))
        return pocket

    def buildPockets(self):
        pockets = []
        corners = list(self.corners)
        getKey = lambda a : math.sqrt(a[0]**2+a[1]**2)
        corners = sorted(corners, key=getKey)
        for i in range(len(self.corners)):
            if not corners:
                break
            x, y = corners.pop(0)  # Pop off the first corner
            pocket = [(x, y)]
            neighbors = [(a, b) for a in range(x-2, x+3) for b in range(y-2, y+3)]  # Grab the nearest corners
            neighbors.remove((x, y))
            for z, w in neighbors:
                if (z, w) in corners:
                    corners.remove((z, w))
                    pocket.extend([(z, w)] + self.buildPockets_recurse((z, w), corners))
            pockets.append(pocket)
        pockets_averaged = []
        for group in pockets:
            avg_x = sum([p[0] for p in group])/len(group)
            avg_y = sum([p[1] for p in group])/len(group)
            pockets_averaged.append((avg_x, avg_y))
        return pockets_averaged

    def b_center_matrix(self):
        r, c = self.b_center_of_area()
        y_difference = (self.rows/2) - r
        x_difference = (self.cols/2) - c
        if x_difference == 0 and y_difference == 0:
            return self.b_matrix
        centered_matrix = np.copy(self.b_matrix)
        for x in range(self.rows):
            for y in range(self.cols):
                if self.b_matrix[x , y] == 1:
                    centered_matrix[x+y_difference, y+x_difference], centered_matrix[x, y] = 1, 0
        return centered_matrix

    def intensityMap(self):
        image_array = self.original_matrix
        image_out = image_array
        rows = self.rows
        cols = self.cols
        for i in range(rows):
            for j in range(cols):
                if image_array[i][j] == 1:
                    neighbors = [image_array[x][y] for x in range(max(i-1, 0), min(i+2, rows)) for y in range(max(0, j-1), min(j+2, cols))]
                    image_out[i][j] = neighbors.count(1)

        return image_out

    def distances(self):
        points = self.grouped_corners
        if len(points) > 1:
            distances = {}
            for i in range(len(points)):
                a = points[i]
                for j in range(i + 1, len(points)):
                    b = points[j]
                    distances[(a, b)] = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            maximum = max(distances, key=distances.get)
            print "Max distance between " + str(maximum) + "= " + str(distances[maximum])
            minimum = min(distances, key=distances.get)
            print "Min distance between " + str(minimum) + "= " + str(distances[minimum])
        else:
            print "Only " + str(len(points)) + " point(s)"

    def getEdgeList(self):
        edgelist = []
        edge_horizont = ndimage.sobel(self.r_s_z_b_c_matrix, 0)
        edge_vertical = ndimage.sobel(self.r_s_z_b_c_matrix, 1)
        magnitude = np.hypot(edge_horizont, edge_vertical)
        for x in range(len(magnitude)):
            for y in range(len(magnitude[x])):
                if magnitude[x][y] != float(0):
                    edgelist.append((x, y))
        return edgelist

    def objectDFS(self,matrix,v): #DFS? v = (r,c,discovered,value)??? matrix v2 has a discovered 2 object
    #NO MATRIX CAN BE A COPY, IF 1 ITS NOT DISCOVERED, JUST CHANGE THE VALUE AFTERWARDS LOL
        object1 = [] #
        stack = []
        r,c = v
        stack.append(v) #BUT WHAT IS V???
        while stack:
            v = stack.pop()
            r,c = v
            #print stack
            if(matrix[r][c]==1):  #means undiscovered #1 == discovered, 0 ==undiscovered
                matrix[r][c] = 0
                object1.append(v)
                neighborhood = [(rn,cn) for rn in range(max(r-1, 0), min(r+2, self.rows)) for cn in range(max(0, c-1), min(c+2, self.cols))]
                neighborhood.remove((r,c))
                for v1 in neighborhood:
                    stack.append(v1)
        #print matrix
        return object1,matrix,len(object1) #list of all the components of the shape, along with a cleaned up matrix




    def findObjects(self):
        total_ones = self.area_
        objects = []
        matrix = np.array(self.original_matrix)
        for r in range(self.rows):
            for c in range(self.cols):
                if matrix[r][c]==1:
                    new_object,new_matrix,ones = self.objectDFS(matrix,(r,c))
                    total_ones = total_ones - ones
                    matrix = new_matrix
                    objects.append(new_object)
                    if(total_ones<=0):
                        return objects
        print " no object"
        return []


    def cleanObjects(self):
        objects = self.objects
        cleaned = []
        for obj in objects:
            if len(obj)> 0.05*min(self.rows,self.cols) and len(obj)>4:
                cleaned.append(obj)
        return cleaned


class Shape(object):

    def __init__(self, shape,obj):
        self.obj = obj
        self.rows, self.cols = shape
        self.area_clean = len(obj)
        self.center = self.centerCoordinates(self.obj)
        self.max_r,self.min_r,self.max_c,self.min_c = self.getSize(self.obj)
        self.height_clean = self.max_r - self.min_r
        self.width_clean = self.max_c - self.min_c
        self.size_clean = self.height_clean * self.width_clean

        self.clean_matrix = self.cleanMatrix()

        self.centered_matrix = self.center_matrix()
        self.centered_obj = self.centered_tuples()

        self.center = self.centerCoordinates(self.centered_obj)

        self.theta = self.axis_of_least_movement()

        self.rotated_matrix = self.rotate()

        self.rotated_obj = self.rotated_tuples()

        self.max_r_rotate,self.min_r_rotate,self.max_c_rotate,self.min_c_rotate = self.getSize(self.rotated_obj)

        self.scaled_matrix = self.pad_scaled_matrix()

        self.obj = self.findObjects()

        #self.scaled_matrix = self.scaleMatrix()
        self.area_scale = len(obj)
        self.height_scale = self.max_r-self.min_r
        self.width_scale = self.max_r - self.min_r

        self.size_scale = self.height_scale * self.width_scale  #THESE ARE ALL

        self.height_to_width = self.height_to_width_ratio()

        self.size_to_area = self.size_to_area_ratio()

        self.shape_corners = self.shape_cornerDetector()

        self.shape_grouped_corners = self.buildPockets()

        self.corner_neighborhood = self.cornerNeighborhood()

    def getSize(self,obj):

        max_r,min_r,max_c,min_c = obj[0][0],obj[0][0],obj[0][1],obj[0][1]


        for r,c in obj:
            if r>max_r:
                max_r = r
            if c>max_c:
                max_c = c
            if r<min_r:
                min_r = r
            if c<min_c:
                min_c = c
        return max_r,min_r,max_c,min_c


    def cleanMatrix(self):
        clean_matrix = np.zeros((self.rows,self.cols))
        for r,c in self.obj:
            clean_matrix[r][c] = 1
        return clean_matrix.astype(int)


    def shape_cornerDetector(self):
        matrix = self.clean_matrix
        corners = []
        for r,c in self.obj:
            neighbors = [matrix[rn][cn] for rn in range(max(r-1, 0), min(r+2, self.rows)) for cn in range(max(0, c-1), min(c+2, self.cols))]
            if neighbors.count(0)>4:
                corners.append((r,c))
        return corners

    def objectDFS(self,matrix,v): #DFS? v = (r,c,discovered,value)??? matrix v2 has a discovered 2 object
    #NO MATRIX CAN BE A COPY, IF 1 ITS NOT DISCOVERED, JUST CHANGE THE VALUE AFTERWARDS LOL
        object1 = [] #
        stack = []
        r,c = v
        stack.append(v) #BUT WHAT IS V???
        while stack:
            v = stack.pop()
            r,c = v
            #print stack
            if(matrix[r][c]==1):  #means undiscovered #1 == discovered, 0 ==undiscovered
                matrix[r][c] = 0
                object1.append(v)
                neighborhood = [(rn,cn) for rn in range(max(r-1, 0), min(r+2, self.rows)) for cn in range(max(0, c-1), min(c+2, self.cols))]
                neighborhood.remove((r,c))
                for v1 in neighborhood:
                    stack.append(v1)
        #print matrix
        return object1,matrix,len(object1) #list of all the components of the shape, along with a cleaned up matrix




    def findObjects(self):
        total_ones = self.scaled_matrix.sum()
        objects = []
        matrix = np.array(self.scaled_matrix)
        for r in range(self.scaled_matrix.shape[0]):
            for c in range(self.scaled_matrix.shape[1]):
                if matrix[r][c]==1:
                    new_object,new_matrix,ones = self.objectDFS(matrix,(r,c))
                    total_ones = total_ones - ones
                    matrix = new_matrix
                    objects.append(new_object)
                    if(total_ones<=0):
                        return objects[0]
        print " no object"
        return []


    def buildPockets_recurse(self, t, corners):
        x,y = t[0], t[1]
        if len(corners) == 0:
            return []
        pocket = []
        neighbors = [(a, b) for a in range(x-2, x+3) for b in range(y-2, y+3)]  # Grab the nearest corners
        neighbors.remove((x, y))
        for z, w in neighbors:
            if (z, w) in corners:

                corners.remove((z, w))
                pocket.extend([(z, w)] + self.buildPockets_recurse((z, w), corners))
        return pocket

    def buildPockets(self):
        pockets = []
        corners = list(self.shape_corners)
        getKey = lambda a : math.sqrt(a[0]**2+a[1]**2)
        corners = sorted(corners, key=getKey)
        for i in range(len(self.shape_corners)):
            if not corners:
                break
            x, y = corners.pop(0)  # Pop off the first corner
            pocket = [(x, y)]
            neighbors = [(a, b) for a in range(x-2, x+3) for b in range(y-2, y+3)]  # Grab the nearest corners
            neighbors.remove((x, y))
            for z, w in neighbors:
                if (z, w) in corners:
                    corners.remove((z, w))
                    pocket.extend([(z, w)] + self.buildPockets_recurse((z, w), corners))
            pockets.append(pocket)
        pockets_averaged = []
        for group in pockets:
            avg_x = int(round(sum([p[0]*1. for p in group])/len(group)))
            avg_y = int(round(sum([p[1]*1. for p in group])/len(group)))
            pockets_averaged.append((avg_x, avg_y))
        return pockets_averaged


    def cornerNeighborhood(self):
        corners = self.shape_grouped_corners
        img = self.clean_matrix
        neighborhoods = []
        for r,c in corners:
            try:
                neighborhood = np.array([img[a][b] for a in range(r-2,r+3) for b in range(c-2,c+3)]).reshape((5,5))
            except IndexError:
                continue
            neighborhoods.append(neighborhood)
        return neighborhoods


    def centerCoordinates(self,group):  # Returns estimated center of object
        return int(round(sum([p[0]*1. for p in group])/len(group))), int(round(sum([p[1]*1. for p in group])/len(group)))


    def zoom(self):
        return np.copy(self.rotated_matrix[self.min_r_rotate : self.max_r_rotate + 1, self.min_c_rotate : self.max_c_rotate + 1])

    def center_matrix(self):
        r,c = self.center
        big_r, big_c = self.rows/2, self.cols/2
        y_dist = int(big_r - r)
        x_dist = int(big_c - c)
        if math.sqrt(big_r**2*big_c**2)-math.sqrt(r**2 * c**2) < (self.rows*self.cols)/20.:
            return self.clean_matrix
        centered_matrix = np.zeros([self.rows, self.cols])
        for row,col in self.obj:
            centered_matrix[row+y_dist, col+x_dist] = 1
        return centered_matrix.astype(int)

    def centered_tuples(self):
        r, c = self.center
        big_r, big_c = self.rows / 2, self.cols / 2
        if math.sqrt(big_r ** 2 * big_c ** 2) - math.sqrt(r ** 2 * c ** 2) < (self.rows * self.cols) / 20.:
            return self.obj
        y_dist = int(big_r - r)
        x_dist = int(big_c - c)
        centered_tuples = []
        for row,col in self.obj:
            centered_tuple = row+y_dist, col+x_dist
            centered_tuples.append(centered_tuple)
        return centered_tuples


    def scale_matrix_(self):
        zoomed_img = self.zoom()
        # for line in self.clean_matrix:
        #     print ' '.join(map(str,line))
        # for line in zoomed_img:
        #     print ' '.join(map(str,line))
        prev_zoom = zoomed_img
        while zoomed_img.shape[0] < self.rows and zoomed_img.shape[1] < self.cols:
                prev_zoom = zoomed_img
                zoomed_img = self.scale(zoomed_img)
        if zoomed_img.shape[0] > self.rows or zoomed_img.shape[1] > self.cols:
            return prev_zoom
        return zoomed_img

    def pad_scaled_matrix(self):
        canvas = np.zeros((self.rows, self.cols)).astype(int)
        scaled_matrix= self.scale_matrix_()
        y_diff = int((self.rows - scaled_matrix.shape[0])/2)
        x_diff = int((self.cols - scaled_matrix.shape[1])/2)
        for r in range(scaled_matrix.shape[0]):
            for c in range(scaled_matrix.shape[1]):
                canvas[r+y_diff,c+x_diff] = scaled_matrix[r,c]
        return canvas

    def scale(self, arr):
        a = np.kron(arr, np.ones((2,2)))
        return a.astype(int)

    def axis_of_least_movement(self):
        a = 0.0
        b = 0.0
        c = 0.0
        img = self.centered_matrix
        r_, c_ = self.center
        for r in range(self.rows):
            for c in range(self.cols):
                a += (r - r_) * (c - c_) * img[r,c]
                b += (r - r_) ** 2 * img[r,c]
                c += (c - c_) ** 2 * img[r,c]
        out = 2. * a / (b - c)
        return 0.5 * math.atan(out)

    def rotate(self):
        return ndimage.rotate(self.centered_matrix, -math.degrees(self.theta), reshape = False).astype(int)

    def size_to_area_ratio(self):
        try:
            return self.area_clean*1./self.size_clean
        except ZeroDivisionError:
            return 1

    def height_to_width_ratio(self):
        try:
            return self.height_clean*1./self.width_clean
        except ZeroDivisionError:
            return 1

    def hamming_distance(self, img, arr2):
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if img[r][c] == arr2[r][c]:
                    shared += 1
        return shared/self.size_clean


    def rotated_tuples(self):
        rotated_coords = []
        for rows in range(self.rows):
            for cols in range(self.cols):
                if self.rotated_matrix[rows,cols] == 1:
                    coord = rows,cols
                    rotated_coords.append(coord)
        return rotated_coords