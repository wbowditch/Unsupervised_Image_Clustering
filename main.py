#!/usr/bin/env python
import sys
import os
from Image import *
import profile
from scipy.optimize import fsolve,brute,fmin
import numpy as np
import time
import threading
from multiprocessing import Process


folders = ['ones','twos','threes','fours','fives','sixs','sevens','eights','nines','zeros']
path = "/Users/williambowditch/Desktop/data"

names = ['eight146.txt','five16.txt','four219.txt','one302.txt','seven137.txt','nine214.txt','six155.txt','three199.txt','two280.txt','zero141.txt']
"""  when you find the image and are trying ot scale it, you calculate the four corners and then find a buffer such that the array is square """
def load_queries(path):
    queries = []

    for file in os.listdir(path):
        if not file.startswith('.'):
            print file
            x = Image(path+"/"+file)
            queries.append(x)
    return queries


def load_database(path,bigTest):
    database = []
    print os.getcwd()
    count = 0
    if bigTest:
        for file in os.listdir(path):
            if file in folders:
                print file
                for name in os.listdir(path+'/'+file):
                    if not name.startswith('.'):
                        if name in names:
                            continue
                        else:
                            count+=1
                            x = Image(path+'/'+file+'/'+name)
                            database.append(x)
                            if count%100 ==0:
                                break
                                print 'working...',count

    else:
        for file in os.listdir(path):
            if not file.startswith('.'):
                print file
                x = Image(path+"/"+file)
                database.append(x)

    return database

def getDecisions(database,queries):
    score = []
    for image in queries:
        print image.file_name
        arr,c = image.decisionTree(database,k=9)
        print arr,c
        score.append(c)
    print score


if __name__ == '__main__':
    name,query_path,database_path,k = sys.argv
    database = load_database(database_path,True)
    queries = load_queries(query_path)
    getDecisions(database,queries,k=int(k))

