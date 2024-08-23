from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model


# 定义U-Net模型，U - Net是一个常用于图像分割的卷积神经网络架构。它由编码器（下采样部分）和解码器（上采样部分）组成，结合了高级和低级特征。


def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # 编码器
    c1 = Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')(inputs)
    c1 = Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')(c1)

    p1 = MaxPooling2D((2, 2))(c1)  # 0.5
    c2 = Conv2D(128, (3, 3), (1, 1), 'same', activation='relu')(p1)
    c2 = Conv2D(128, (3, 3), (1, 1), 'same', activation='relu')(c2)

    p2 = MaxPooling2D((2, 2))(c2)  # 0。25
    c3 = Conv2D(256, (3, 3), (1, 1), 'same', activation='relu')(p2)
    c3 = Conv2D(256, (3, 3), (1, 1), 'same', activation='relu')(c3)

    p3 = MaxPooling2D((2, 2))(c3)  # 0.125
    c4 = Conv2D(512, (3, 3), (1, 1), 'same', activation='relu')(p3)
    c4 = Conv2D(512, (3, 3), (1, 1), 'same', activation='relu')(c4)

    # 编码器底层
    p4 = MaxPooling2D((2, 2))(c4)  # 0.0625
    c5 = Conv2D(1024, (3, 3), (1, 1), 'same', activation='relu')(p4)
    c5 = Conv2D(1024, (3, 3), (1, 1), 'same', activation='relu')(c5)

    # 解码器
    u1 = Conv2DTranspose(512, (2, 2), (2, 2), 'same', activation='relu')(c5)  # 0.125
    """这里之所以用(2,2)的卷积核，可以联系最大池化的过程"""
    u1 = concatenate([u1, c4])
    c6 = Conv2D(512, (3, 3), (1, 1), 'same', activation='relu')(u1)
    c6 = Conv2D(512, (3, 3), (1, 1), 'same', activation='relu')(c6)

    u2 = Conv2DTranspose(256, (2, 2), (2, 2), 'same', activation='relu')(c6)  # 0.25
    u2 = concatenate([u2, c3])
    c7 = Conv2D(256, (3, 3), (1, 1), 'same', activation='relu')(u2)
    c7 = Conv2D(256, (3, 3), (1, 1), 'same', activation='relu')(c7)

    u3 = Conv2DTranspose(128, (2, 2), (2, 2), 'same', activation='relu')(c7)  # 0.5
    u3 = concatenate([u3, c2])
    c8 = Conv2D(128, (3, 3), (1, 1), 'same', activation='relu')(u3)
    c8 = Conv2D(128, (3, 3), (1, 1), 'same', activation='relu')(c8)

    u4 = Conv2DTranspose(64, (2, 2), (2, 2), 'same', activation='relu')(c8)  # 1
    u4 = concatenate([u4, c1])
    c9 = Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')(u4)
    c9 = Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')(c9)

    outputs = Conv2D(1, (1, 1), (1, 1), 'same', activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


