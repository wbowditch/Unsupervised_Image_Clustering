#Generate symmetrical image
import random
import sys
width = 20
height = 20

def generate_horizontal_symmetry(file_num,confidence):
    file = open("horizontal_symmetry_database/horizontal_symmetry_image"+str(file_num)+"."+str(confidence)+".txt",'w')

    for _ in range(height/2):
        ones = ['1' if random.uniform(0,1)<confidence else '0' for x in range(width) ]
        ones_str = ' '.join(ones)
        file.write(ones_str + '\n')
    for _ in range(height/2):
        zeros = ['0' if random.uniform(0,1)<confidence else '1' for x in range(width)]
        zeros_str = ' '.join(zeros)
        file.write(zeros_str + '\n')

    file.close()


def generate_vertical_symmetry(file_num,confidence):
    file = open("vertical_symmetry_database/vertical_symmetry_image"+str(file_num)+"."+str(confidence)+".txt",'w')

    for _ in range(height):
        ones = ['1' if random.uniform(0,1)<confidence else '0' for x in range(width/2) ]
        zeros = ['0' if random.uniform(0,1)<confidence else '1' for x in range(width/2)]
        line = ' '.join(ones+zeros)
        file.write(line + '\n')
    file.close()

def generate_chessboard_symmetry(file_num,confidence):
    file = open("chessboard_symmetry_database/chessboard_symmetry_image"+str(file_num)+"."+str(confidence)+".txt",'w')

    for _ in range(height/2):
        ones = ['1' if random.uniform(0,1)<confidence else '0' for x in range(width/2) ]
        zeros = ['0' if random.uniform(0,1)<confidence else '1' for x in range(width/2)]
        line = ' '.join(ones+zeros)
        file.write(line + '\n')
    for _ in range(height/2):
        ones = ['1' if random.uniform(0,1)<confidence else '0' for x in range(width/2) ]
        zeros = ['0' if random.uniform(0,1)<confidence else '1' for x in range(width/2)]
        line = ' '.join(zeros+ones)
        file.write(line + '\n')
    file.close()

for x in range(int(sys.argv[1])):
    generate_chessboard_symmetry(x,float(sys.argv[2]))








