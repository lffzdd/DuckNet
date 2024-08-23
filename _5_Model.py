import gc

import tensorflow as tf

from _1_ImageLoader import load_imagedata
from _2_ConvBlock import duckV2Block
from _3_DuckNet import create_duck_model
from _4_ImageAugmentation_DiceLost import image_augmentation, dice_loss
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score
import numpy as np

print('设备：', tf.config.list_physical_devices())
# %% 初始化
seed = 58800
filters = 17

image_train, mask_train = load_imagedata(images=1000)
image_train, image_test, mask_train, mask_test = train_test_split(image_train, mask_train, test_size=0.1, shuffle=True,
                                                                  random_state=seed)
image_train, image_valid, mask_train, mask_valid = train_test_split(image_train, mask_train, test_size=0.111,
                                                                    shuffle=True, random_state=seed)

model = create_duck_model(filters)

model.compile(optimizer='rmsprop', loss=dice_loss)
# %%
np.savez('image_data.npz',
         image_train=image_train, mask_train=mask_train,
         image_valid=image_valid, mask_valid=mask_valid,
         image_test=image_test, mask_test=mask_test)

# %%
seed = 58800
filters = 17

image_data = np.load('image_data.npz')

image_train = image_data['image_train']
mask_train = image_data['mask_train']
image_valid = image_data['image_valid']
mask_valid = image_data['mask_valid']
image_test = image_data['image_test']
mask_test = image_data['mask_test']

# model = create_duck_model(filters)
#
# model.compile(optimizer='rmsprop', loss=dice_loss)
# %% 训练并保存模型
step = 0
EPOCHS = 600
learning_rate = 1e-4
dataset_type = 'kvasir'
model_type = 'DuckNet'
min_loss_for_saving = 0.2  # 当dice相似度大于0.8时首次保存模型
# %%
progress_path = 'ProgressFull/' + dataset_type + '_progress_csv_' + model_type + '_filters_' + str(
    filters) + '_' + '.csv'
progressfull_path = 'ProgressFull/' + dataset_type + '_progress_' + model_type + '_filters_' + str(
    filters) + '_' + '.txt'
plot_path = 'ProgressFull/' + dataset_type + '_progress_plot_' + model_type + '_filters_' + str(filters) + '_' + '.png'
model_path = 'ModelSaveTensorFlow/' + dataset_type + '/' + model_type + '_filters_' + str(filters) + '_'

step = 165
min_loss_for_saving = 0.0857
model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss})

for epoch in range(496, EPOCHS):
    # 记录每个周期训练过程中的数据
    print(f'第{epoch + 1}个周期')
    print(f'学习率:{learning_rate}')
    step += 1

    image_augmented, mask_augmented = image_augmentation(image_train, mask_train)

    csv_logger = CSVLogger(progress_path, append=True, separator=';')

    model.fit(image_augmented, mask_augmented, batch_size=4, epochs=1, verbose=1, callbacks=[csv_logger],
              validation_data=(image_valid, mask_valid))

    pred_valid = model.predict(image_valid, verbose=0)
    loss_valid = dice_loss(mask_valid, pred_valid).numpy()
    print('验证集dice:', loss_valid)

    pred_test = model.predict(image_test, verbose=0)
    loss_test = dice_loss(mask_test, pred_test).numpy()
    print('测试集dice:', loss_test)

    with open(progressfull_path, 'a') as f:
        f.write(f'周期 {epoch}\n验证集dice:{loss_valid}\t测试集dice:{loss_test}\n')

    if loss_valid < min_loss_for_saving:
        min_loss_for_saving = loss_valid
        print('保存模型，验证集dice为：', loss_valid)
        model.save(model_path)
        with open(progressfull_path, 'a') as f:
            f.write('保存模型，验证集dice为：' + str(loss_valid))

    del image_augmented
    del mask_augmented

    gc.collect()

# %% 加载模型并计算衡量指标
print('加载模型-' + model_path)
model = tf.keras.models.load_model(model_path, custom_objects={'diec_loss': dice_loss})  # 自定义函数不会自动保存，需要显示
# %%
print('=====模型预测=====')
pred_train = model.predict(image_train, batch_size=4)
pred_test = model.predict(image_test, batch_size=4)
pred_valid = model.predict(image_valid, batch_size=4)
print('=====预测完成=====')

mask_train = np.ndarray.flatten(np.array(mask_train, dtype=bool))
mask_test = np.ndarray.flatten(np.array(mask_test, dtype=bool))
mask_valid = np.ndarray.flatten(np.array(mask_valid, dtype=bool))
pred_train = np.ndarray.flatten(pred_train > 0.5)
pred_test = np.ndarray.flatten(pred_test > 0.5)
pred_valid = np.ndarray.flatten(pred_valid > 0.5)

print('=====衡量指标=====')

# 所有交并比
dice_train = f1_score(mask_train, pred_train)
dice_test = f1_score(mask_test, pred_test)
dice_valid = f1_score(mask_valid, pred_valid)
print('===dice完成===')

# 前景交并比
miou_train = jaccard_score(mask_train, pred_train)
miou_test = jaccard_score(mask_test, pred_test)
miou_valid = jaccard_score(mask_valid, pred_valid)
print('===miou完成===')

# 即预测是正例的结果中，确实是正例的比例。
precision_train = precision_score(mask_train, pred_train)
precision_test = precision_score(mask_test, pred_test)
precision_valid = precision_score(mask_valid, pred_valid)
print('===precision完成===')

# 即所有正例的样本中，被找出的比例
recall_train = recall_score(mask_train, pred_train)
recall_test = recall_score(mask_test, pred_test)
recall_valid = recall_score(mask_valid, pred_valid)
print('===recall完成===')

# 即所有分类正确的样本占全部样本的比例
accuracy_train = accuracy_score(mask_train, pred_train)
accuracy_test = accuracy_score(mask_test, pred_test)
accuracy_valid = accuracy_score(mask_valid, pred_valid)
print('===accuracy完成===')

# %%
final_file = 'results_' + model_type + '_' + str(filters) + '_' + dataset_type + '.txt'

with open(final_file, 'a') as f:
    f.write(dataset_type + '\n\n')
    f.write(f'dice_train:{str(dice_train)}\tdice_valid:{str(dice_valid)}\tdice_test:{str(dice_test)}\n')
    f.write(f'miou_train:{str(miou_train)}\tmiou_valid:{str(miou_valid)}\tmiou_test:{str(miou_test)}\n')
    f.write(
        f'precision_train:{str(precision_train)}\tprecision_valid:{str(precision_valid)}precision_test:{str(precision_test)} \n')
    f.write('recall_train: ' + str(recall_train) + ' recall_valid: ' + str(recall_valid) + ' recall_test: ' + str(
        recall_test) + '\n')
    f.write(
        'accuracy_train: ' + str(accuracy_train) + ' accuracy_valid: ' + str(accuracy_valid) + ' accuracy_test: ' + str(
            accuracy_test) + '\n\n\n\n')

# %% 看看预测图片
from PIL import Image

image_data = np.load('image_data.npz')

image_train = image_data['image_train']
mask_train = image_data['mask_train']
image_valid = image_data['image_valid']
mask_valid = image_data['mask_valid']
image_test = image_data['image_test']
mask_test = image_data['mask_test']

pred_train = model.predict(image_train, batch_size=4)
pred_test = model.predict(image_test, batch_size=4)
pred_valid = model.predict(image_valid, batch_size=4)

image_train_img = (image_train[0] * 255).astype('uint8')
image_train_img = Image.fromarray(image_train_img)
pred_train_img = Image.fromarray((pred_train[0, :, :, 0] * 255).astype('uint8'))
image_train_img.show()
pred_train_img.show()
image_train_img.save('img[0].png')
pred_train_img.save('pred[0].png')
