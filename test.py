# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("pc1.png", 0)

img = cv2.GaussianBlur(img, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)  # 这里对最后一个参数使用了经验型的值
lines = cv2.HoughLinesP(edges,3,np.pi/180,30,minLineLength=80,maxLineGap=70)

lines1 = lines[:,0,:]#提取为二维
for x1, y1, x2, y2 in lines1[:]:
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)



# result=img[b[0]:b[2],b[1]:b[3]]
cv2.imshow('p',img)
# result = img
# for line in lines[0]:
#     rho = line[0]  # 第一个元素是距离rho
#     theta = line[1]  # 第二个元素是角度theta
#     print
#     rho
#     print
#     theta
#     if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
#         # 该直线与第一行的交点
#         pt1 = (int(rho / np.cos(theta)), 0)
#         # 该直线与最后一行的焦点
#         pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
#         # 绘制一条白线
#         cv2.line(result, pt1, pt2, (255,255,0),1)
#     else:  # 水平直线
#         # 该直线与第一列的交点
#         pt1 = (0, int(rho / np.sin(theta)))
#         # 该直线与最后一列的交点
#         pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
#         # 绘制一条直线
#         cv2.line(result, pt1, pt2, (255,255,0), 1)
#
# cv2.imshow('Canny', edges)
# cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()