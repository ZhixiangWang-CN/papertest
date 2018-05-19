import os
import cv2
import numpy as np

# -*- coding: utf-8 -*-
  # 文件夹目录
#os.chdir( "./scan")
os.chdir( "./photo")
files_name = os.listdir()
print(files_name)

n=len(files_name)


def rrname():
    p = range(50, 97, 1)
    for i in range(n):
        test=str(p[i])+str('.jpg')
        print(test)
        #file = path_scan + '/' + files_name[i]
        #print(file)
        os.rename(files_name[i],test)
        #print(test)

def rname():
    for i in range(n):
        test=str(i)+str('.jpg')
        print(test)
        #file = path_scan + '/' + files_name[i]
        #print(file)
        os.rename(files_name[i],test)
if __name__ == '__main__':
    rname()