import numpy as np

import _1_ImageLoader
from TEST._2_1_Unet import unet_model
from TEST._2_4_augmentation import image_augmentation
from sklearn.model_selection import train_test_split

# %%
image_train, mask_train = _1_ImageLoader.load_imagedata(images=64)
image_train, image_valid, mask_train, mask_valid = train_test_split(image_train, mask_train, test_size=0.1,
                                                                    random_state=42)

print(image_train.shape)
# %%
model = unet_model()

train_generator = image_augmentation(image_train, mask_train)

print("========================================模型概要========================================")
model.summary()
print("========================================训练开始========================================")
model.fit(train_generator, steps_per_epoch=len(image_train) // 32,
          validation_data=(image_valid, mask_valid))  # 每个周期用来验证

# %%
# 最终评估
loss, accuracy = model.evaluate(image_valid, mask_valid)


# %%
# 预测和可视化结果
def plot_sample(image_data, mask_data, index=None):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    image_data *= 255
    pred = model.predict(image_valid)
    if index is None:
        index = np.random.randint(0, len(image_data))

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    image = image_data[index].squeeze().astype(np.uint8)
    mask = mask_data[index].squeeze().astype(np.uint8)
    pred = pred[index].squeeze().astype(np.uint8)

    ax[0].imshow(image).set_title(f'第{index}张：\t原图').contour(mask, colors='r', levels=[0.5])
    ax[1].imshow(mask).set_title('掩码图')
    ax[2].imshow(pred).set_title('预测图').contour(mask, colors='r', levels=[0.5])
