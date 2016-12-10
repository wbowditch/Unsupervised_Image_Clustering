#WILL BOWDITCH
#BINARY OBJECT FEATURE EXTRACTION
#Playing around with area, center of area, shared squares between matrices
#and axis of least second movement.
import math

def create_array(confidence):
    file = open("category_database/horizontal_symmetry_database/horizontal_symmetry_image0."+confidence+".txt",'r')
    array = []
    for line in file:
        array.append([int(x) for x in line.split(" ")])
    return array

array10 = create_array("1.0")
array9 = create_array("0.9")
array8 = create_array("0.8")
array7 = create_array("0.7")
array3 = create_array("0.3")
array2 = create_array("0.2")
array1 = create_array("0.1")
array0 = create_array("0.0")

fulldate = []
fulldate.append(array0)
fulldate.append(array1)
fulldate.append(array2)
fulldate.append(array3)
fulldate.append(array7)
fulldate.append(array9)
fulldate.append(array10)

# file2 = open("horizontal_symmetry_database/horizontal_symmetry_image0.0.9.txt",'r')
# array2 = []
# for line in file2:
#     array2.append([int(x) for x in line.split(" ")])
# #print array



def I(r,c):
    return array[r][c]


#indicates relative size of the object
def area(arr):
    area = 0
    for r in range(len(arr)):
        for c in range(len(arr[0])):
            area+= arr[r][c]
    return area

#print area()

#These correspond to the row and column coordinate of the center of the ith object.
def center_of_area(arr):
    area_ = area(arr)
    r_=0
    c_=0
    for r in range(len(arr)):
        for c in range(len(arr[0])):
            r_+= r*arr[r][c]
            c_+= c*arr[r][c]
    a = 1./area_
    r_ = r_*a
    c_ = c_*a

    return c_,r_
#raw comparison of amt of 1s and 0s b/w two arrays
def squares_shared(arr1,arr2):
    shared = 0.0
    total = len(arr1)*len(arr1[0])

    for r in range(len(arr1)):
        for c in range(len(arr1[0])):
            if arr1[r][c] == arr2[r][c]:
                shared+=1
    return shared/total

"""print "SQUARES SHARED WITH ARRAY 10 (HORIZONTAL SYMMETRY, TOP HALF 1 BOTTOM HALF 0)"
print squares_shared(array10,array9)
print squares_shared(array10,array8)
print squares_shared(array10,array7)
print squares_shared(array10,array3)
print squares_shared(array10,array2)
print squares_shared(array10,array1)
print squares_shared(array10,array0)
"""
def mostCommon(arr):
    queryImage = arr[-1]
    max = 0.0
    maxArr = []
    queryArray = arr[:-1]
    for array in queryArray:
        if squares_shared(queryImage, array) > max:
            max = squares_shared(queryImage, array)
            maxArr = array
    return maxArr, max

arr, v = mostCommon(fulldate)
print("squares shared: %.2f" % (v))
#for line in arr:
 #   print line

def axis_of_least_second_movement(arr):
    a = 0.0
    b = 0.0
    c = 0.0
    c_,r_ = center_of_area(arr)
    for r in range(len(arr)):
        for c in range(len(arr[0])):
            a+= (r - r_)*(c-c_)*arr[r][c]
            b+= (r-r_)**2*arr[r][c]
            c+= (c-c_)**2*arr[r][c]
    out = 2.*a/(b - c)
    return 0.5*math.atan(out)

print "AXIS"
print axis_of_least_second_movement(array10)
print axis_of_least_second_movement(array9)
print axis_of_least_second_movement(array8)
print axis_of_least_second_movement(array7)
print axis_of_least_second_movement(array3)
print axis_of_least_second_movement(array2)
print axis_of_least_second_movement(array1)
print axis_of_least_second_movement(array0)
print
print "AREA CENTER"
print center_of_area(array10)
print center_of_area(array9)
print center_of_area(array8)
print center_of_area(array7)
print center_of_area(array3)
print center_of_area(array2)
print center_of_area(array1)
print center_of_area(array0)




