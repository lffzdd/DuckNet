import cv2

image = cv2.imread('img_1.png', cv2.IMREAD_GRAYSCALE)
ret, segmented = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# 自适应均值阈值
adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,0 )
# 自适应高斯加权平均值阈值
adaptive_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,0 )

# 显示结果
cv2.imshow('image',image)
cv2.imshow('common',segmented)
cv2.imshow('adaptive mean', adaptive_mean)
cv2.imshow('adaptive Gaussian', adaptive_gaussian)

cv2.waitKey(0) # 暂停，避免图像一闪而过
cv2.destroyAllWindows()

cv2.imwrite('2-ConvBlock/2-image.jpg',image)
cv2.imwrite('2-ConvBlock/2-common.jpg',segmented)
cv2.imwrite('2-ConvBlock/2-adaptive mean.jpg', adaptive_mean)
cv2.imwrite('2-ConvBlock/2-adaptive Gaussian.jpg', adaptive_gaussian)