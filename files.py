import os
import cv2
import numpy as np
# -*- coding: utf-8 -*-


# print(files_name)
# a=len(files_name)
def filepath_scan(num):
    path_scan = "../paper/scan"  # 文件夹目录
    files_name = os.listdir(path_scan)
    file = path_scan+'//'+files_name[num]
    return file
def files_name_scan():
    path_scan = "../paper/scan"  # 文件夹目录
    files_name = os.listdir(path_scan)
    return files_name

def filepath_photo(num):
    path_photo = "../paper/photo"  # 文件夹目录
    files_name = os.listdir(path_photo)
    file = path_photo+'/'+files_name[num]
    return file
def files_name_photo():
    path_photo = "../paper/photo"  # 文件夹目录
    files_name_p = os.listdir(path_photo)
    return files_name_p

# print(file)

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

#test filepath whether is ok

# img=cv_imread(filepath_photo(9))
# print(filepath_photo(9))
# cv2.imshow("1",img)
# cv2.waitKey()