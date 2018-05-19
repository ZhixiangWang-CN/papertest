#!/usr/bin/python3
# 2017.11.11 01:44:37 CST
# 2017.11.12 00:09:14 CST
"""
使用Sift特征点检测和匹配查找场景中特定物体。
"""
import os
import sys
import test2
import files
import cv2
import numpy as np
# -*- coding: utf-8 -*-
import cv2
import numpy as np
MIN_MATCH_COUNT = 4


# path = ".\scan"
# files_name=os.listdir(path)



def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img


def findbest(num,mmgood,maxnum):
    file = files.filepath_scan(num)
    img2 = cv_imread(file)
    imgname2 = img2
    ## (1) prepare data


    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    # (2) Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    ## (3) Create flann matcher
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

    ## (4) Detect keypoints and compute keypointer descriptors
    kpts1, descs1 = sift.detectAndCompute(gray1,None)
    kpts2, descs2 = sift.detectAndCompute(gray2,None)

    ## (5) knnMatch to get Top2
    matches = matcher.knnMatch(descs1, descs2, 2)
    # Sort by their distance.
    matches = sorted(matches, key = lambda x:x[0].distance)

    ## (6) Ratio test, to get good matches.
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]
    lg=len(good)
    # print("good:%s"%good)
    print("正在检测%s:%d"%(ffc[i],lg))

    if lg>mmgood:
        mmgood=lg
        maxnum=num

    # print(len(good))

# canvas = img2.copy()
if __name__ == '__main__':
    mmgood = 0
    maxnum = 0
    jiance=int(sys.argv[1])
    max=test2.pipei5(jiance)
    # path1 = "./photo"
    files_name_p = files.filepath_photo(jiance)
    print("待检测目标为：%s"%files_name_p)
    file1 = files.filepath_photo(jiance)
    img1 = cv_imread(file1)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ffc = files.files_name_scan()
    for i in max:
        findbest(i,mmgood,maxnum)
    ffc=files.files_name_scan()
    # print("原图像为：%s"%files_name1[jiance])
    print("最近匹配为%s:"%ffc[maxnum])
## (7) find homography matrix
## 当有足够的健壮匹配点对（至少4个）时
# if len(good)>MIN_MATCH_COUNT:
#     ## 从匹配中提取出对应点对
#     ## (queryIndex for the small object, trainIndex for the scene )
#     src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     ## find homography matrix in cv2.RANSAC using good match points
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     ## 掩模，用作绘制计算单应性矩阵时用到的点对
#     #matchesMask2 = mask.ravel().tolist()
#     ## 计算图1的畸变，也就是在图2中的对应的位置。
#     h,w = img1.shape[:2]
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#     ## 绘制边框
#     cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))


# ## (8) drawMatches
# matched = cv2.drawMatches(img1,kpts1,canvas,kpts2,good,None)#,**draw_params)

# ## (9) Crop the matched region from scene
# h,w = img1.shape[:2]
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.perspectiveTransform(pts,M)
# perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
# found = cv2.warpPerspective(img2,perspectiveM,(w,h))

## (10) save and display
# cv2.imwrite("matched.png", matched)
# cv2.imwrite("found.png", found)
# cv2.imshow("matched", matched);
# cv2.imshow("found", found);
# cv2.waitKey();cv2.destroyAllWindows()