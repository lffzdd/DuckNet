import glob
import os.path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_PATH = os.path.join('.' + '/' + 'assets', 'images')
# MASK_PATH = os.path.join('.' + '/' + 'assets', 'masks')
print(IMAGE_PATH)


def load_imagedata(img_w: int = 352, img_h: int = 352, images: int = -1):
    """
    加载图像数据
    :param img_w: 图像宽度
    :param img_h: 图像高度
    :param images: 图像数量
    :return: 原图像数据，掩码图像数据
    """
    image_list = glob.glob(IMAGE_PATH + '/' + '*.jpg')

    image_num = len(image_list)
    # 输入图像数量
    if images == -1:
        images = image_num
    image_list = image_list[:images]

    image_train = np.zeros((images, img_h, img_w, 3), dtype=np.float32)
    mask_train = np.zeros((images, img_h, img_w), dtype=np.uint8)
    for index, image in tqdm(enumerate(image_list), desc='导入图像', total=images):
        mask = image.replace('images', 'masks')

        # 统一尺寸，掩码图像使用了LANCZOS插值
        image = Image.open(image).resize((img_w, img_h))
        mask = Image.open(mask).resize((img_w, img_h), resample=Image.Resampling.LANCZOS)

        # 转化成numpy数组
        image = np.array(image)
        mask = np.array(mask)

        # 归一化
        image = image / 255
        for h in range(img_h):
            for w in range(img_w):
                if mask[h, w] >= 127:
                    mask[h, w] = 1
                else:
                    mask[h, w] = 0

        image_train[index] = image
        mask_train[index] = mask

    # 使输出与输入维度一致
    mask_train = np.expand_dims(mask_train, axis=-1)

    return image_train, mask_train

