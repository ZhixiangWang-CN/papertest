import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(".\pc1.png")
# cv2.imshow("1",img)
sp = img.shape
print (sp)
#gaosibuller
# img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.pyrMeanShiftFiltering(img, 25, 10)
print("111")

ray=[]
# cv2.CvColor(img,ray,cv2.COLOR_BGR2GRAY)
ray=cv2.cvtColor( img,cv2.COLOR_BGR2GRAY )
canny = cv2.Canny(ray, 10, 100)
# cv2.imshow("2",img)
# cv2.imshow("3",canny)
# box=cv2.boundingRect(ray)
b=cv2.boundingRect(ray)
# cv2.imshow('p',ray)
print(b)
#houh
# lines = cv2.HoughLinesP(canny,3,np.pi/180,30,minLineLength=60,maxLineGap=10)
lines = cv2.HoughLines(canny, 1, np.pi / 180, 30)
# print(lines)
lines1 = lines[:,0,:]#提取为二维
middlex = []
middley = []
i=0
# for x1, y1, x2, y2 in lines1[:]:
#     m1=(x1+x2)/2
#     m2=(y1+y2)/2
#     i=i+1
#     middlex.append(m1)
#     middley.append(m2)
#
# n=len(middlex)
# print(middlex)
# print(middley)
# line3=[]
# for i in range(n):
#     for j in range(n):
#         a = middley[i]
#         b = middley[j]
#         if (abs(b-a)<10):
#
#             if b>a:
#                 line3.append(j)
# print(line3)
# mylist=[]
# line3 = list(set(line3))
# print(line3)
# conterx=[]
# contery=[]

# for x1,y1,x2,y2 in lines1[:]:
#     c1=(x2+x1)/2
#     c2=(y2+y1)/2
#     conterx.append(c1)
#     contery.append(c2)
# n=len(conterx)
# print("x=",n)
# c3=[]
# for i in range(n):
#     for j in range(n):
#         d=contery[i]
#         if d-contery[i]<10:
#             c3.append(lines1[i])
# print(c3)
# n=len(c3)
# print("c3=",n)
# print(lines1)
for x1, y1, x2, y2 in lines1[:]:
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
# result=img[b[0]:b[2],b[1]:b[3]]
cv2.imshow('p',img)

#grabcut
#
# mask = np.zeros(img.shape[:2],np.uint8)
# # 背景模型
# bgdModel = np.zeros((1,65),np.float64)
# # 前景模型
# fgdModel = np.zeros((1,65),np.float64)
#
# rect = (0,0,sp[0],sp[1])
# # 使用grabCut算法
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
#
# # img = img*mask2[:,:,np.newaxis]#使用蒙板来获取前景区域
#
#
#
# cv2.imshow('p',img)
# result = cv2.grabCut( ray,rect, Mat& bgdModel, Mat& fgdModel, int iterCount, int mode )
# [u1:d1, u2:d2]
# img.crop((0, 0, 200, 200))
# cv2.imshow("4",result)
# plt.subplot(122),plt.imshow(img,)
# plt.xticks([]),plt.yticks([])
cv2.waitKey(0)