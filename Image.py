#All text files will be read into this class
import math
import numpy as np
class Image(object):

    def __init__(self,file_name,rows=0,cols=0):
        self.matrix = self._create_matrix(file_name)
        self.rows = rows if rows !=0 else len(self.matrix)
        self.cols = cols if cols !=0 else len(self.matrix[0])
        self.size = len(self.matrix)*len(self.matrix[0])
        #self.four_corners()
        self.north = self.north()
        self.south = self.south()
        self.east = self.east()
        self.west = self.west()
        #print self.matrix
        #self.matrix = self.mean_average_blur()

    def _create_matrix(self,name):
        file = open(name,'r')
        array = []
        for line in file:
            #print line
            array.append([int(x) for x in line.split(" ")])
        return np.array(array)

    def reshape(self):
        rows = self.rows
        cols = self.cols
        x,y = self.center_of_area()
        print x,y

        s = int(round(self.area()**0.5))+2
        x1 = 0 if x-s<0 else x-s
        x2 = rows if x+s>rows else x+s
        y1 = 0 if y-s<0 else y-s
        y2 = cols if y+s>cols else y+s

        print s
        #print x-s,x+s
        #print y-s,y+s
        output = self.matrix
        output = output[x1:x2]
        for i in range(len(output)):

            output[i] = output[i][y1:y2]
        print 'hello'
        print output
        return output
        # print y - s,y+s
        # z = self.matrix[x-s:x+s][y-s:y+s]
        # for q in z:
        #     print q
        # print
        # return self.matrix[x-s:x+s][y-s:y+s]


    def mean_average_blur(self,alpha = 10):
        image_array = self.matrix
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
        return image_output

    def __len__(self):
        return self.size

    def rows(self):
        return self.rows

    def cols(self):
        return self.cols

    def area(self):
        area = 0
        for r in range(self.rows):
            for c in range(self.cols):
                area+= self.matrix[r][c]
        return area


    def center_of_area(self):
        area_ = self.area()
        r_=0
        c_=0
        for r in range(self.rows):
            for c in range(self.cols):
                r_+= r*self.matrix[r][c]
                c_+= c*self.matrix[r][c]
        a = 1./area_
        r_ = r_*a
        c_ = c_*a

        return int(round(c_)),int(round(r_))


# x = xcostheta y sin theta

    def axis_of_least_second_movement(self):
        a = 0.0
        b = 0.0
        c = 0.0
        c_,r_ = self.center_of_area()
        for r in range(self.rows):
            for c in range(self.cols):
                a+= (r - r_)*(c-c_)*self.matrix[r][c]
                b+= (r-r_)**2*self.matrix[r][c]
                c+= (c-c_)**2*self.matrix[r][c]
        out = 2.*a/(b - c)
        return 0.5*math.atan(out)


    def hamming_distance(self,arr2):
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.matrix[r][c] == arr2[r][c]:
                    shared+=1
        return shared/self.size

    def scale(self,scale_x,scale_y): #Kronecker product
        #scale_x = self.rows/sub_rows
        #scale_y = self.cols/sub_cols

        a = np.kron(self.matrix, np.ones((scale_x,scale_y)))
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

    def four_corners(self):
        return_array = []
        for i in range (self.rows):
            for j in range (self.cols):
                if self.matrix[i][j] == 1:
                    self.north = np.array([i,j])
                    #return_array.append([i,j])#north
                    break

        for i in range (self.cols):
            for j in range (self.rows):
                if self.matrix[i][j] == 1:
                    self.west = np.array([i,j])
                    #return_array.append([i,j])#west
                    break
        for i in range (self.rows-1, 0, -1):
            for j in range(self.cols-1, 0, -1):
                if self.matrix[i][j] == 1:
                    self.south = np.array([i,j])
                    #return_array.append([i,j])#south
                    break
        for i in range (self.cols-1, 0, -1):
            for j in range (self.rows-1, 0, -1):
                if self.matrix[i][j] == 1:
                    self.east = np.array([i,j])
                    #return_array.append([i,j])#east
                    break

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



    def findCorners(self,window_size,k, thresh):
        """
        Finds and returns list of corners and new image with corners drawn
        :param img: The original image
        :param window_size: The size (side length) of the sliding window
        :param k: Harris corner constant. Usually 0.04 - 0.06
        :param thresh: The threshold above which a corner is counted
        :return:
        """
        #Find x and y derivatives
        img = self.matrix
        dy, dx = np.gradient(img)
        I = np.identity(img.shape[0])
        #dx= dx * I
        #dy = dy * I
        Ixx = dx**2
        Ixy = dy*dx
        Iyy = dy**2
        height = img.shape[0]
        width = img.shape[1]

        cornerList = []
        newImg = img.copy()
        #color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
        offset = window_size/2

        #Loop through image and find our corners
        print "Finding Corners..."
        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                #Calculate sum of squares
                windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
                windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
                windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
                Sxx = windowIxx.sum()
                Sxy = windowIxy.sum()
                Syy = windowIyy.sum()

                #Find determinant and trace, use to get corner response
                det = (Sxx * Syy) - (Sxy**2)
                trace = Sxx + Syy
                r = det - k*(trace**2)
                #r = min(Sxx,Syy)

                #If corner response is over threshold, color the point and add to corner list
                if r > thresh:
                    #print x, y, r
                    cornerList.append([x, y, r])
                    # color_img.itemset((y, x, 0), 0)
                    # color_img.itemset((y, x, 1), 0)
                    # color_img.itemset((y, x, 2), 255)
        return cornerList


    def cornerDetector(self):
        corners = []
        image_array = self.matrix
        rows = self.rows
        cols = self.cols
        for i in range(rows):
            for j in range(cols):
                if image_array[i][j] ==1:
                    neighbors = [image_array[x][y] for x in range(max(i-1,0),min(i+2,rows)) for y in range(max(0,j-1),min(j+2,cols))]
                    #print neighbors
                    if neighbors.count(0)>4:
                        corners.append((i,j))

        return corners



    def cornersPocket(self):
        pockets







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



















