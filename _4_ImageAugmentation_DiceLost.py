import numpy as np
import albumentations as albu
import tensorflow.keras.backend as K

aug_img = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter((0.6, 1.6), 0.2, 0.1, 0.01, True),
    albu.Affine((0.5, 1.5), (-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22.5), always_apply=True)
])


def image_augmentation(image_train, mask_train):
    image_augmented = []
    mask_augmented = []
    for i in range(len(image_train)):
        aug = aug_img(image=image_train[i], mask=mask_train[i])
        image_augmented.append(aug['image'])
        mask_augmented.append(aug['mask'])

    return np.array(image_augmented), np.array(mask_augmented)


def dice_loss(img_true, img_pred):
    # 转换成float32格式便于计算
    img_true = K.cast(img_true, 'float32')
    img_pred = K.cast(img_pred, 'float32')

    # 不用flatten一样的
    # img_true=K.flatten(img_true)
    # img_pred=K.flatten(img_pred)

    intersection = K.sum(img_true * img_pred)
    union = K.sum(img_true) + K.sum(img_pred)

    smooth = 1e-6
    dice = (intersection * 2. + smooth) / (union + smooth)
    return 1 - dice
