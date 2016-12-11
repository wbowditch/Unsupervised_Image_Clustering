#All text files will be read into this class
import math
import numpy as np
class Image(object):

    def __init__(self,file_name,rows=0,cols=0):
        self.matrix = self._create_matrix(file_name)
        self.rows = rows if rows !=0 else len(self.matrix)
        self.cols = cols if cols !=0 else len(self.matrix[0])
        self.size = len(self.matrix)*len(self.matrix[0])
        self.quadrants = []
        self.theta = 1.
        self.scale = 1.
        #print self.matrix
        #self.matrix = self.mean_average_blur()

    def _create_matrix(self,name):
        file = open(name,'r')
        array = []
        for line in file:
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

    #blur filter
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
        return np.sum(self.matrix)


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
        self.theta = 0.5*math.atan(out)
        return 0.5*math.atan(out)


    def hamming_distance(self,arr2):
        shared = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.matrix[r][c] == arr2[r][c]:
                    shared+=1
        return shared/self.size

    def four_corners(self):
        return_array = []
        for i in range (self.rows):
            for j in range (self.cols):
                if self.matrix[i][j] == 1:
                    return_array.append([i,j])#north
                    break
        for i in range (self.cols):
            for j in range (self.rows):
                if self.matrix[i][j] == 1:
                    return_array.append([i,j])#west
                    break
        for i in range (self.rows, 0, -1):
            for j in range(self.cols, 0. -1):
                if self.matrix[i][j] == 1:
                    return_array.append([i,j])#south
                    break
        for i in range (self.cols, 0, -1):
            for j in range (self.rows, 0, -1):
                if self.matrix[i][j] == 1:
                    return_array.append([i,j])#east
                    break