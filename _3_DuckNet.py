from keras.layers import Input, Conv2D, add, UpSampling2D
from keras.models import Model
from _2_ConvBlock import duckV2Block, resnetBlock


def create_duck_model(filters, input_size=(352, 352, 3), out_size_classes=1):
    inputs = Input(input_size)
    d = duckV2Block(inputs, filters)

    # 下采样
    p1 = Conv2D(filters * 2, 2, strides=2, padding='same')(inputs)
    p2 = Conv2D(filters * 4, 2, strides=2, padding='same')(p1)
    p3 = Conv2D(filters * 8, 2, strides=2, padding='same')(p2)
    p4 = Conv2D(filters * 16, 2, strides=2, padding='same')(p3)
    p5 = Conv2D(filters * 32, 2, strides=2, padding='same')(p4)

    d_p1 = Conv2D(filters * 2, 2, strides=2, padding='same')(d)
    d_p1 = add([p1, d_p1])
    d_p1 = duckV2Block(d_p1, filters * 2)

    d_p2 = Conv2D(filters * 4, 2, strides=2, padding='same')(d_p1)
    d_p2 = add([p2, d_p2])
    d_p2 = duckV2Block(d_p2, filters * 4)

    d_p3 = Conv2D(filters * 8, 2, strides=2, padding='same')(d_p2)
    d_p3 = add([p3, d_p3])
    d_p3 = duckV2Block(d_p3, filters * 8)

    d_p4 = Conv2D(filters * 16, 2, strides=2, padding='same')(d_p3)
    d_p4 = add([p4, d_p4])
    d_p4 = duckV2Block(d_p4, filters * 16)

    d_p5 = Conv2D(filters * 32, 2, strides=2, padding='same')(d_p4)
    d_p5 = add([p5, d_p5])
    d_p5 = resnetBlock(d_p5, filters * 32)
    d_p5 = resnetBlock(d_p5, filters * 32)
    d_p5 = resnetBlock(d_p5, filters * 16)
    d_p5 = resnetBlock(d_p5, filters * 16)

    # 上采样
    u_p4 = UpSampling2D()(d_p5)
    u_p4 = add([d_p4, u_p4])
    u_p4 = duckV2Block(u_p4, filters * 8)

    u_p3 = UpSampling2D()(u_p4)
    u_p3 = add([d_p3, u_p3])
    u_p3 = duckV2Block(u_p3, filters * 4)

    u_p2 = UpSampling2D()(u_p3)
    u_p2 = add([d_p2, u_p2])
    u_p2 = duckV2Block(u_p2, filters * 2)

    u_p1 = UpSampling2D()(u_p2)
    u_p1 = add([d_p1, u_p1])
    u_p1 = duckV2Block(u_p1, filters)

    u = UpSampling2D()(u_p1)
    u = add([d, u])
    u = duckV2Block(u, filters)

    output = Conv2D(1, (1, 1), activation='sigmoid')(u)

    model = Model(inputs=inputs, outputs=output)
    return model
