import cv2
import matplotlib.pyplot as plt

img = cv2.imread('vessel2.png', 0)

# 灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 高斯滤波 卷积 3 * 3
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
# x梯度
xgrad = cv2.Sobel(img_blur, cv2.CV_16SC1, 1, 0)
# y梯度
ygrad = cv2.Sobel(img_blur, cv2.CV_16SC1, 0, 1)
# 使用梯度参数进行边缘检测 阈值 50 ~ 150
edge1 = cv2.Canny(xgrad, ygrad, 50, 150)
# 直接用高斯滤波结果进行边缘检测 阈值 50 ~ 150
edge2 = cv2.Canny(img_blur, 50, 150)
cv2.imshow('origin image', img)
cv2.imshow('edge image', edge1)
cv2.imshow('edge image2', edge2)
cv2.waitKey()