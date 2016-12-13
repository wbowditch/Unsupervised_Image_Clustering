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
        self.rows = self.original_matrix.shape[0]
        self.cols = self.original_matrix.shape[1]
        self.size = len(self.original_matrix) * len(self.original_matrix[0])
        self.has_inverted_matrix = self.invert_or_not()
        if self.has_inverted_matrix:
            self.original_matrix = self.invert_matrix()
        self.area_ = self.area()
        # BLUR Image, get area and coordinates of blurred image object
        self.b_matrix = self.mean_average_blur()
        self.b_area_ = self.b_area()
        self.b_center = self.b_center_of_area()
        self.b_radians = self.b_axis_of_least_second_movement()
        self.b_c_matrix = self.b_center_matrix()

        # Points
        self.east_point = self.east()
        self.west_point = self.west()
        self.north_point = self.north()
        self.south_point = self.south()

        # Zoom
        self.keep = self.keep_or_not()
        self.z_b_c_matrix = self.zoom()  # old = self.z_r_b_matrix
        self.z_rows = self.z_b_c_matrix.shape[0]
        self.z_cols = self.z_b_c_matrix.shape[1]
        self.scale_cols = self.cols / self.z_cols
        self.scale_rows = self.rows / self.z_rows

        # Scale
        self.s_z_b_c_matrix = self.scale()
        self.r_s_z_b_c_matrix = self.rotate_blurred_matrix()  # old = self.s_z_r_b_matrix

        # Extra Features
        self.corners = self.cornerDetector()
        self.grouped_corners = self.buildPockets()
        self.neighborhoods = self.corner_neighborhood()
        self.edges = self.getEdgeList()
        self.edge_groups = self.edges_neighborhood()


    def _create_matrix(self,name):  # Reads image from textfile into numpy array
        file = open(name, 'r')
        array = []
        for line in file:
            array.append([int(x) for x in line.split(" ")])
        return np.array(array)

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




    def rotate_blurred_matrix(self):
        rads = self.b_radians
        return ndimage.rotate(self.s_z_b_c_matrix, -math.degrees(rads),reshape=False)


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
                     ):  # return k closets neighbors
        euclideanDistance = lambda a,b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        score = {image: 0 for image in database_images}
        for image in database_images:
            points = 0
            if abs(self.area_ - image.area_) < area_sigma:
                points += 2
            else:
                points -= 1

            if abs(self.b_area_ - image.b_area_) < b_area_sigma:
                points += 2
            else:
                points -= 1

            if euclideanDistance(self.b_center,image.b_center) < center_diff_sigma:
                points += 2
            else:
                points -= 1

            if abs(self.b_radians - image.b_radians) == rads_sigma:
                points += 2
            else:
                points -= 1

            if abs(self.scale_cols - image.scale_cols) < scale_cols_sigma:
                points += 2
            else:
                points -= 1

            if abs(self.scale_rows - image.scale_rows) < scale_rows_sigma:
                points += 2
            else:
                points -= 1

            if self.hamming_distance4(image.r_s_z_b_c_matrix) > hamming_simga1:
                points += 2
            else:
                points -= 1

            if self.hamming_distance4(image.r_s_z_b_c_matrix) > hamming_simga2:
                points += 2
            else:
                points -= 1

            if self.hamming_distance4(image.r_s_z_b_c_matrix) > hamming_simga3:
                points += 3

            if self.hamming_distance4(image.r_s_z_b_c_matrix) > hamming_simga4:
                points += 4
            c = 0
            for neighborhood1 in self.neighborhoods:
                U1 = svd(neighborhood1,compute_uv=False)
                for neighborhood2 in image.neighborhoods:
                    U2 = svd(neighborhood2,compute_uv=False)
                    diff = abs(sum(U1 - U2))
                    if diff == 0:
                        points += 2
                        c += 1
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
        if self.north_point < 0 or self.south_point > self.rows or self.west_point < 0 or self.east_point > self.cols or self.south_point-self.north_point > self.rows or self.east_point-self.west_point > self.cols:
            return False
        else:
            return True

    def calculate_ratios(self):
        dy, dx = np.gradient(self.original_matrix)
        north_south = self.south - self.north
        east_west = self.east - self.west
        ns_ew = float(north_south)/float(east_west)
        return dy,dx

    def cornerDetector(self):
        corners = []
        image_array = self.r_s_z_b_c_matrix
        rows = self.rows
        cols = self.cols
        for i in range(rows):
            for j in range(cols):
                if image_array[i][j] == 1:
                    neighbors = [image_array[x][y] for x in range(max(i-1, 0), min(i+2, rows)) for y in range(max(0, j-1), min(j+2, cols))]
                    if neighbors.count(0) > 4:
                        corners.append((i, j))
        self.corners = corners
        return corners

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
        if x_difference < self.cols/10. and y_difference < self.rows/10.:
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