import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def generator():
    """
    生成器是迭代器的一种 \n
    生成器可以快速建立迭代器 \n
    zip()将可迭代对象打包成迭代器
    :return:
    """
    a = [1, 2, 3, 4, 5, 6]
    a = zip(a)
    for i in a:
        print(i)  # >>>:(1,) (2,) (3,) (4,) (5,) (6,)

    def gen1():  # 生成器函数
        yield 1  # 相当于for i in range(1,4)
        yield 2
        yield 3

    a = gen1()
    for i in a:
        print(i)

    a = (x for x in range(5))  # 生成器表达式
    for i in a:
        print("(x for x in range(5)):", i)

    a = (x for x in range(5))
    print(next(a))
    print(next(a))
    print(next(a))
    print(next(a))
    print(next(a))

    a = zip((x for x in range(5)), (y ** 2 for y in range(5)))
    print(next(a))
    print(next(a))
    print(next(a))
    print(next(a))

    def gen2():
        yield [0, 1]
        yield [0, 2]
        yield [0, 3]

    for i in gen2():
        print("gen2():", i)

    def gen3():
        yield [9, 1]
        yield [9, 2]
        yield [9, 3]

    a = zip(gen2(), gen3())
    print(next(a))
    print(next(a))
    print(next(a))
    # print(next(a))  # >>>报错


def image_augmentation(image_train, mask_train):
    data_gen_args = dict(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.5,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    seed = 1
    image_data_gen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_data_gen.fit(image_train, augment=True, seed=seed)
    mask_datagen.fit(mask_train, augment=True, seed=seed)

    # Create separate generators for images and masks using the same generator object
    image_generator = image_data_gen.flow(image_train, batch_size=32, seed=seed)
    mask_generator = image_data_gen.flow(mask_train, batch_size=32, seed=seed)

    train_generator = zip(image_generator, mask_generator)
    return train_generator


def test_augmentation():
    from PIL import Image
    import numpy as np

    image = Image.open('img.png')
    image = np.array(image)
    image = np.expand_dims(image, 0)

    args = {
        'rotation': 180,
        'shear': 30,
        'zoom': 7  # [lower, upper] = [1-zoom_range, 1+zoom_range]，
        # 参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作。

    }
    image_data_gen = ImageDataGenerator(zoom_range=args['zoom'])

    image_generator = image_data_gen.flow(image, batch_size=1, shuffle=False)
    for img in image_generator:
        img = Image.fromarray(img[0].astype(np.uint8))
        img.show()

        img.save('3-augmentation/' + 'zoom_range=' + str(args['zoom']) + '.png')
        break

# test_augmentation()
