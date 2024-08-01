import math

import numpy as np
from scipy.signal import convolve2d

# 原始图像矩阵
I = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 12, 90, 89, 86, 87, 82, 0],
    [0, 10, 12, 88, 85, 83, 84, 0],
    [0, 9, 15, 12, 84, 84, 88, 0],
    [0, 12, 14, 10, 82, 88, 89, 0],
    [0, 11, 17, 16, 12, 88, 90, 0],
    [0, 10, 16, 15, 17, 89, 88, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

# Sobel卷积核Gx
Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Gx = Gx[::-1, ::-1]

Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
Gy = Gy[::-1, ::-1]
O_h = convolve2d(I, Gx, mode='valid')
O_v = convolve2d(I, Gy, mode='valid')

print(O_h)
print(O_v)

grad = np.sqrt(O_h ** 2 + O_v ** 2)
grad_direct = np.arctan(O_v / O_h)
grad_direct = np.degrees(grad_direct)

# 梯度大小
# print(grad.astype(np.int32))

# 梯度方向
np.set_printoptions(precision=2, suppress=True)  # 设置精度并抑制科学计数法
print((O_v / O_h).astype(np.float32))
print(grad_direct)
