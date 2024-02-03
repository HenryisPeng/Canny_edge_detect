'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#读者在使用反色变换前请安装numpy库和pillow库


def image_inverse(x):#定义反色变换函数
    value_max = np.max(x)
    y = value_max - x
    return y

if __name__ == '__main__':
    gray_img = np.asarray(Image.open(r'1.tif').convert('L'))
    #Image.open是打开图片，变量为其地址，
    inv_img = image_inverse(gray_img) #将原图形作为矩阵传入函数中，进行反色变换

    fig = plt.figure()#绘图
    ax1 = fig.add_subplot(121)#解释一下121，第一个1是一行，2是是两列，第二个1是第一个图
    ax1.set_title('Orignal')
    ax1.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Transform')
    ax2.imshow(inv_img, cmap='gray', vmin=0, vmax=255)

    plt.show()


import cv2
import random
import imutils
import numpy as np

# 彩色图像每个像素值是[x,y,z], 灰度图像每个像素值便是一个np.uint8
image = cv2.imread('1.tif')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图像变为灰度图像（RGB彩色变灰色）

# 图像大小调整
ori_h, ori_w = image.shape[:2]  # 获得原图像长宽
height, width = gray_img.shape[:2]  # 获得灰度图像长宽
image = cv2.resize(image, (int(ori_w / ori_h * 400), 400), interpolation=cv2.INTER_CUBIC)  # 对图像大小变换且做三次插值
gray_img = cv2.resize(gray_img, (int(width / height * 400), 400), interpolation=cv2.INTER_CUBIC)  # 对图像大小变换且做三次插值

# a<0 and b=0: 图像的亮区域变暗，暗区域变亮
a, b = -0.5, 0
new_img1 = np.ones((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)  # 初始化一个新图像做变换，且大小同灰度图像大小
for i in range(new_img1.shape[0]):
    for j in range(new_img1.shape[1]):
        new_img1[i][j] = gray_img[i][j] * a + b  # 原始图像*a+b

# a>1: 增强图像的对比度,图像看起来更加清晰
a, b = 1.5, 20
new_img2 = np.ones((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
for i in range(new_img2.shape[0]):
    for j in range(new_img2.shape[1]):
        if gray_img[i][j] * a + b > 255:
            new_img2[i][j] = 255
        else:
            new_img2[i][j] = gray_img[i][j] * a + b

# a<1: 减小了图像的对比度, 图像看起来变暗
a, b = 0.5, 0
new_img3 = np.ones((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
for i in range(new_img3.shape[0]):
    for j in range(new_img3.shape[1]):
        new_img3[i][j] = gray_img[i][j] * a + b

# a=1且b≠0, 图像整体的灰度值上移或者下移, 也就是图像整体变亮或者变暗, 不会改变图像的对比度
a, b = 1, -50
new_img4 = np.ones((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
for i in range(new_img4.shape[0]):
    for j in range(new_img4.shape[1]):
        pix = gray_img[i][j] * a + b
        if pix > 255:
            new_img4[i][j] = 255
        elif pix < 0:
            new_img4[i][j] = 0
        else:
            new_img4[i][j] = pix

# a=-1, b=255, 图像翻转
new_img5 = 255 - gray_img

cv2.imshow('origin', imutils.resize(image, 800))
cv2.imshow('gray', imutils.resize(gray_img, 800))
cv2.imshow('a<0 and b=0', imutils.resize(new_img1, 800))
cv2.imshow('a>1 and b>=0', imutils.resize(new_img2, 800))
cv2.imshow('a<1 and b>=0', imutils.resize(new_img3, 800))
cv2.imshow('a=1 and b><0', imutils.resize(new_img4, 800))
cv2.imshow('a=-1 and b=255', imutils.resize(new_img5, 800))
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

'''

from PIL import Image
import numpy as np

# 打开tif图像
image = Image.open("1.tif")

# 转换为灰度图像
gray_image = image.convert("L")

# 将灰度图像转换为NumPy数组
gray_array = np.array(gray_image)

# 进行灰度变换，可以根据需要自定义变换函数
def grayscale_transform(pixel_value):
    # 这里示例将纯黑的像素值变为128，可以根据需要进行调整
    if pixel_value == 0:
        return 128
    else:
        return pixel_value

# 应用灰度变换
transformed_array = np.vectorize(grayscale_transform)(gray_array)

# 创建新的灰度图像
transformed_image = Image.fromarray(transformed_array, "L")

# 保存变换后的图像
transformed_image.save("transformed.tif")

# 显示原始和变换后的图像
image.show()
transformed_image.show()
