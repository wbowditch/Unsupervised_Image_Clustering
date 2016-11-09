#All text files will be read into this class
import math

class Image(object):

    def __init__(self,file_name,rows=0,cols=0):
        self.matrix = self._create_matrix(file_name)
        self.rows = rows if rows !=0 else len(self.matrix)
        self.cols = cols if cols !=0 else len(self.matrix[0])
        self.size = len(self.matrix)*len(self.matrix[0])


    def _create_matrix(name):
        file = open("test_images/"+name+".txt",'r')
        array = []
        for line in file:
            array.append([int(x) for x in line.split(" ")])
        return array

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
        area_ = self.area(self.matrix)
        r_=0
        c_=0
        for r in range(self.rows):
            for c in range(self.cols):
                r_+= r*self.matrix[r][c]
                c_+= c*self.matrix[r][c]
        a = 1./area_
        r_ = r_*a
        c_ = c_*a

        return c_,r_



    def axis_of_least_second_movement(self):
        a = 0.0
        b = 0.0
        c = 0.0
        c_,r_ = self.center_of_area(self.matrix)
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

