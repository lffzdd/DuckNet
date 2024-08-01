from keras.layers import Input, Conv2D, add
from _2_ConvBlock import duckV2Block


def create_duck_model(filters, input_size=(352, 352, 3), out_size_classes=1):
    inputs = Input(input_size)

    # 下采样
    p1 = Conv2D(filters * 2, 2, strides=2, padding='same')(inputs)
    p2 = Conv2D(filters * 4, 2, strides=2, padding='same')(p1)
    p3 = Conv2D(filters * 8, 2, strides=2, padding='same')(p2)
    p4 = Conv2D(filters * 16, 2, strides=2, padding='same')(p3)
    p5 = Conv2D(filters * 32, 2, strides=2, padding='same')(p4)

    d1 = duckV2Block(inputs, filters)

    d_p1 = Conv2D(filters * 2, 2, strides=2, padding='same')(d1)
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
    d_p5 = duckV2Block(d_p5, filters * 32)


