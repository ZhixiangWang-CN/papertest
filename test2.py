import cv2
import numpy as np
import files
from matplotlib import pyplot as plt
import os
import array
# 最简单的以灰度直方图作为相似比较的实现
# path = ".\scan" #文件夹目录
# files_name=os.listdir(path)
# len_files=len(files_name)
decsize=(200,200)
def classify_gray_hist(image1,image2,size = (256,256)):
 # 先计算直方图
 # 几个参数必须用方括号括起来
 # 这里直接用灰度图计算直方图，所以是使用第一个通道，
 # 也可以进行通道分离后，得到多个通道的直方图
 # bins 取为16
 cv2.namedWindow("img1", 2);
 cv2.resizeWindow("img1", 640, 480);
 cv2.namedWindow("img2", 0);
 cv2.resizeWindow("img2", 640, 480);
 # cv2.namedWindow("1.img", 300,400)
 # cv2.namedWindow("2.img", 300, 400)
 image1 = cv2.resize(image1,size)
 image2 = cv2.resize(image2,size)
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])
 # 可以比较下直方图
 plt.plot(range(256),hist1,'r')
 plt.plot(range(256),hist2,'b')
 plt.show()
 # 计算直方图的重合度
 degree = 0
 for i in range(len(hist1)):
  if hist1[i] != hist2[i]:
   degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
  else:
   degree = degree + 1
 degree = degree/len(hist1)
 return degree

# 计算单通道的直方图的相似值
def calculate(image1,image2):
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])
  # 计算直方图的重合度
 degree = 0
 for i in range(len(hist1)):
  if hist1[i] != hist2[i]:
   degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
  else:
   degree = degree + 1
 degree = degree/len(hist1)
 return degree

# 通过得到每个通道的直方图来计算相似度
def classify_hist_with_split(image1,image2,size = (256,256)):
 # 将图像resize后，分离为三个通道，再计算每个通道的相似值
 image1 = cv2.resize(image1,size)
 image2 = cv2.resize(image2,size)
 sub_image1 = cv2.split(image1)
 sub_image2 = cv2.split(image2)
 sub_data = 0
 for im1,im2 in zip(sub_image1,sub_image2):
  sub_data += calculate(im1,im2)
 sub_data = sub_data/3
 return sub_data

# 平均哈希算法计算
def classify_aHash(image1,image2):
 image1 = cv2.resize(image1,decsize)
 image2 = cv2.resize(image2,decsize)
 gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
 hash1 = getHash(gray1)
 hash2 = getHash(gray2)
 return Hamming_distance(hash1,hash2)

def classify_pHash(image1,image2):
 image1 = cv2.resize(image1,(32,32))
 image2 = cv2.resize(image2,(32,32))
 gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
 # 将灰度图转为浮点型，再进行dct变换
 dct1 = cv2.dct(np.float32(gray1))
 dct2 = cv2.dct(np.float32(gray2))
 # 取左上角的8*8，这些代表图片的最低频率
 # 这个操作等价于c++中利用opencv实现的掩码操作
 # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
 dct1_roi = dct1[0:32,0:32]
 dct2_roi = dct2[0:32,0:32]
 hash1 = getHash(dct1_roi)
 hash2 = getHash(dct2_roi)
 return Hamming_distance(hash1,hash2)

# 输入灰度图，返回hash
def getHash(image):
 avreage = np.mean(image)
 hash = []
 for i in range(image.shape[0]):
  for j in range(image.shape[1]):
   if image[i,j] > avreage:
    hash.append(1)
   else:
    hash.append(0)
 return hash


# 计算汉明距离
def Hamming_distance(hash1,hash2):
 num = 0
 for index in range(len(hash1)):
  if hash1[index] != hash2[index]:
   num += 1
 return num

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

# if __name__ == '__main__':
def pipei5(num):
 # path = ".\photo"
 # file = path + '\\' + files_name[num]
 file=files.filepath_photo(num)
 img1 = cv_imread(file)
 # cv2.imshow("img3",img1)
 # img2 = cv2.imread('2.jpg')
 # cv2.imshow("img2",img2)
 # degree = classify_gray_hist(img1,img2)
 # degree = classify_hist_with_split(img1,img2)
 # degree = classify_aHash(img1,img2)
 # degree = classify_pHash(img1,img2)
 mindegree=100000
 min_i=[0]
 ff=files.files_name_scan()
 len_files=len(ff)
 degreelist=[[0 for x in range(2)] for y in range(len_files)]
 for i in range(len_files):
   file=files.filepath_scan(i)
   img2=cv_imread(file)
   if img2 is not None:

    degree = classify_aHash(img1, img2)
    print("正在检测%s:%d" % (ff[i],degree))
    degreelist[i][0]=degree
    degreelist[i][1] =i


    # if degree<mindegree:
    #  mindegree=degree
    #  min_i=i
   else:

    print("图片未加载成功")
    break

 degreelist.sort()

 print("最佳匹配为:")
 min=[0,0,0,0,0]
 for j in range(5):
  min[j]=degreelist[j][1]
  print(ff[min[j]])
 print(min)
 return(min)
 # print (degree)
 # cv2.waitKey(0)