from keras.layers import Conv2D, BatchNormalizationV2, add

kernel_initializer = 'he_uniform'


def spatiallyBlock(x, filters, kernel_size=3):
    """
    空间可分离卷积块，降低计算复杂度

    :param x: 输入数据
    :param filters: 卷积核数量
    :param kernel_size: 卷积核大小
    :return: 输出
    """
    x = Conv2D(filters, (1, kernel_size), padding='same', activation='relu', kernel_initializer=kernel_initializer)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (kernel_size, 1), padding='same')(x)
    x = BatchNormalizationV2(axis=-1)(x)

    return x


def midScopeBlock(x, filters):
    """
    中范围卷积块，采用1、2的空洞卷积
    :param x: 输入数据
    :param filters: 卷积核数量
    :return: 输出
    """
    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
               dilation_rate=1)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
               dilation_rate=2)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    return x


def wideScopeBlock(x, filters):
    """
    大范围卷积块，采用1、2、3的空洞卷积
    :param x: 输入数据
    :param filters: 卷积核数量
    :return: 输出
    """
    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
               dilation_rate=1)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
               dilation_rate=2)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
               dilation_rate=3)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    return x


def resnetBlock(x, filters, dilation_rate=1):
    x = Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_initializer,
               dilation_rate=dilation_rate)(x)

    # x_res = doubleBlock(x, filters, dilation_rate)
    x_res = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
                   dilation_rate=dilation_rate)(x)
    x_res = BatchNormalizationV2(axis=-1)(x_res)

    x_res = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
                   dilation_rate=dilation_rate)(x_res)
    x_res = BatchNormalizationV2(axis=-1)(x_res)

    x = add([x, x_res])
    x = BatchNormalizationV2(axis=-1)(x)

    return x


def doubleBlock(x, filters, dilation_rate=1):
    """
    普通的双层卷积块，每层加上了批归一化
    :param x: 输入
    :param filters: 卷积核数量
    :param dilation_rate: 膨胀率
    :return:
    """
    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_initializer,
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    return x


def duckV2Block(x, filters: int, spatially_size: int = 6):
    x = BatchNormalizationV2(axis=-1)(x)

    x1 = wideScopeBlock(x, filters)

    x2 = midScopeBlock(x, filters)

    x3 = resnetBlock(x, filters)

    x4 = resnetBlock(x, filters)
    x4 = resnetBlock(x4, filters)

    x5 = resnetBlock(x, filters)
    x5 = resnetBlock(x5, filters)
    x5 = resnetBlock(x5, filters)

    x6 = spatiallyBlock(x, filters, spatially_size)

    x = add([x1, x2, x3, x4, x5, x6])
    x = BatchNormalizationV2(axis=-1)(x)

    return x
