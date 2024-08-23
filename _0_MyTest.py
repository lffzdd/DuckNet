import tensorflow.keras.models
from tensorflow.keras.layers import Conv2D, BatchNormalization, add, Input, UpSampling2D
from tensorflow.keras.models import Model


def spatiallyBlock(x, filters, kernel_size=3, name=None):
    x = Conv2D(filters, (1, kernel_size), padding='same', activation='relu', kernel_initializer='he_uniform',
               name=name + '_conv1')(x)
    x = BatchNormalization(axis=-1, name=name + '_bn1')(x)

    x = Conv2D(filters, (kernel_size, 1), padding='same', name=name + '_conv2')(x)
    x = BatchNormalization(axis=-1, name=name + '_bn2')(x)

    return x


def midScopeBlock(x, filters, name=None):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', dilation_rate=1,
               name=name + '_conv1')(x)
    x = BatchNormalization(axis=-1, name=name + '_bn1')(x)

    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', dilation_rate=2,
               name=name + '_conv2')(x)
    x = BatchNormalization(axis=-1, name=name + '_bn2')(x)

    return x


def wideScopeBlock(x, filters, name=None):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', dilation_rate=1,
               name=name + '_conv1')(x)
    x = BatchNormalization(axis=-1, name=name + '_bn1')(x)

    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', dilation_rate=2,
               name=name + '_conv2')(x)
    x = BatchNormalization(axis=-1, name=name + '_bn2')(x)

    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', dilation_rate=3,
               name=name + '_conv3')(x)
    x = BatchNormalization(axis=-1, name=name + '_bn3')(x)

    return x


def resnetBlock(x, filters, dilation_rate=1, name=None):
    x = Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_initializer='he_uniform',
               dilation_rate=dilation_rate, name=name + '_conv1')(x)
    x_res = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform',
                   dilation_rate=dilation_rate, name=name + '_conv2')(x)
    x_res = BatchNormalization(axis=-1, name=name + '_bn1')(x_res)

    x_res = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform',
                   dilation_rate=dilation_rate, name=name + '_conv3')(x_res)
    x_res = BatchNormalization(axis=-1, name=name + '_bn2')(x_res)

    x = add([x, x_res], name=name + '_add')
    x = BatchNormalization(axis=-1, name=name + '_bn3')(x)

    return x


def duckV2Block(x, filters, spatially_size=6, name=None):
    x = BatchNormalization(axis=-1, name=name + '_bn_in')(x)

    x1 = wideScopeBlock(x, filters, name=name + '_wideScope')

    x2 = midScopeBlock(x, filters, name=name + '_midScope')

    x3 = resnetBlock(x, filters, name=name + '_resnet1')

    x4 = resnetBlock(x, filters, name=name + '_resnet2')
    x4 = resnetBlock(x4, filters, name=name + '_resnet3')

    x5 = resnetBlock(x, filters, name=name + '_resnet4')
    x5 = resnetBlock(x5, filters, name=name + '_resnet5')
    x5 = resnetBlock(x5, filters, name=name + '_resnet6')

    x6 = spatiallyBlock(x, filters, spatially_size, name=name + '_spatially')

    x = add([x1, x2, x3, x4, x5, x6], name=name + '_add')
    x = BatchNormalization(axis=-1, name=name + '_bn_out')(x)

    return x


def create_duck_model(filters, input_size=(352, 352, 3), out_size_classes=1):
    inputs = Input(input_size, name='input_layer')
    d = duckV2Block(inputs, filters, name='initial_duckV2')

    # 下采样
    p1 = Conv2D(filters * 2, 2, strides=2, padding='same', name='p1')(inputs)
    p2 = Conv2D(filters * 4, 2, strides=2, padding='same', name='p2')(p1)
    p3 = Conv2D(filters * 8, 2, strides=2, padding='same', name='p3')(p2)
    p4 = Conv2D(filters * 16, 2, strides=2, padding='same', name='p4')(p3)
    p5 = Conv2D(filters * 32, 2, strides=2, padding='same', name='p5')(p4)

    d_p1 = Conv2D(filters * 2, 2, strides=2, padding='same', name='d_p1')(d)
    d_p1 = add([p1, d_p1], name='d_p1_add')
    d_p1 = duckV2Block(d_p1, filters * 2, name='d_p1_duckV2')

    d_p2 = Conv2D(filters * 4, 2, strides=2, padding='same', name='d_p2')(d_p1)
    d_p2 = add([p2, d_p2], name='d_p2_add')
    d_p2 = duckV2Block(d_p2, filters * 4, name='d_p2_duckV2')

    d_p3 = Conv2D(filters * 8, 2, strides=2, padding='same', name='d_p3')(d_p2)
    d_p3 = add([p3, d_p3], name='d_p3_add')
    d_p3 = duckV2Block(d_p3, filters * 8, name='d_p3_duckV2')

    d_p4 = Conv2D(filters * 16, 2, strides=2, padding='same', name='d_p4')(d_p3)
    d_p4 = add([p4, d_p4], name='d_p4_add')
    d_p4 = duckV2Block(d_p4, filters * 16, name='d_p4_duckV2')

    d_p5 = Conv2D(filters * 32, 2, strides=2, padding='same', name='d_p5')(d_p4)
    d_p5 = add([p5, d_p5], name='d_p5_add')
    d_p5 = resnetBlock(d_p5, filters * 32, name='d_p5_resnet1')
    d_p5 = resnetBlock(d_p5, filters * 32, name='d_p5_resnet2')
    d_p5 = resnetBlock(d_p5, filters * 16, name='d_p5_resnet3')
    d_p5 = resnetBlock(d_p5, filters * 16, name='d_p5_resnet4')

    # 上采样
    u_p4 = UpSampling2D(name='u_p4')(d_p5)
    u_p4 = add([d_p4, u_p4], name='u_p4_add')
    u_p4 = duckV2Block(u_p4, filters * 8, name='u_p4_duckV2')

    u_p3 = UpSampling2D(name='u_p3')(u_p4)
    u_p3 = add([d_p3, u_p3], name='u_p3_add')
    u_p3 = duckV2Block(u_p3, filters * 4, name='u_p3_duckV2')

    u_p2 = UpSampling2D(name='u_p2')(u_p3)
    u_p2 = add([d_p2, u_p2], name='u_p2_add')
    u_p2 = duckV2Block(u_p2, filters * 2, name='u_p2_duckV2')

    u_p1 = UpSampling2D(name='u_p1')(u_p2)
    u_p1 = add([d_p1, u_p1], name='u_p1_add')
    u_p1 = duckV2Block(u_p1, filters, name='u_p1_duckV2')

    u = UpSampling2D(name='u')(u_p1)
    u = add([d, u], name='u_add')
    u = duckV2Block(u, filters, name='u_duckV2')

    output = Conv2D(out_size_classes, (1, 1), activation='sigmoid', name='output_layer')(u)

    model = Model(inputs=inputs, outputs=output)
    return model


# %%
# 创建模型
from tensorflow.keras.callbacks import CSVLogger
from _4_ImageAugmentation_DiceLost import dice_loss, image_augmentation
import matplotlib as plt
import numpy as np
import gc

# 加载你的数据，例如一个输入图像
image_data = np.load('image_data.npz')

image_train = image_data['image_train']
mask_train = image_data['mask_train']
image_valid = image_data['image_valid']
mask_valid = image_data['mask_valid']
image_test = image_data['image_test']
mask_test = image_data['mask_test']
# %%
# model = create_duck_model(17)
# model.compile(optimizer='rmsprop', loss=dice_loss)

model=tensorflow.keras.models.load_model('TEST/0-MyTest/ModelSave/model',custom_objects={'dice_loss':dice_loss})

for epoch in range(40, 45):
    # 记录每个周期训练过程中的数据
    print(f'第{epoch + 1}个周期')
    print(f'学习率:{1e-4}')

    image_augmented, mask_augmented = image_augmentation(image_train, mask_train)
    csv_logger = CSVLogger('TEST/0-MyTest/record.csv', append=True, separator=';')

    model.fit(image_augmented, mask_augmented, batch_size=4, epochs=1, verbose=1, callbacks=[csv_logger],
              validation_data=(image_valid, mask_valid))

    pred_valid = model.predict(image_valid, verbose=0)
    loss_valid = dice_loss(mask_valid, pred_valid).numpy()
    print('验证集dice:', loss_valid)

    pred_test = model.predict(image_test, verbose=0)
    loss_test = dice_loss(mask_test, pred_test).numpy()
    print('测试集dice:', loss_test)

    with open('TEST/0-MyTest/record.txt', 'a') as f:
        f.write(f'周期 {epoch}\n验证集dice:{loss_valid}\t测试集dice:{loss_test}\n')

    gc.collect()

model.save('TEST/0-MyTest/ModelSave/model')
# %%
# 选择感兴趣的中间层输出
layer_outputs = [model.get_layer(name).output for name in [
    'output_layer',
    'p1', 'p2', 'p3', 'p4', 'p5',
    'd_p1_duckV2_bn_out', 'd_p2_duckV2_bn_out', 'd_p3_duckV2_bn_out',
    'd_p4_duckV2_bn_out', 'd_p5_resnet4_bn3',
    'u_p4_duckV2_bn_out', 'u_p3_duckV2_bn_out', 'u_p2_duckV2_bn_out',
    'u_p1_duckV2_bn_out', 'u_duckV2_bn_out'
]]
#%%
# 创建一个新的模型，输入为原始模型的输入，输出为这些中间层的输出

intermediate_model = Model(inputs=model.input, outputs=layer_outputs)
intermediate_output = intermediate_model.predict(image_train,batch_size=4)
print(intermediate_output.shape)

# 获取中间层的输出

# %%
from PIL import Image

# 可视化中间层输出
for i, feature_map in enumerate(intermediate_output):
    plt.figure(figsize=(10, 10))
    plt.title(f"Layer {i + 1} Output")
    plt.imshow(feature_map[0, :, :, 0], cmap='viridis')
    plt.axis('off')
    plt.show()
