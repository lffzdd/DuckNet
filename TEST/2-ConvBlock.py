import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical  # kera3中才有
from PIL import Image

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 验证是否正确加载数据
Image.fromarray(x_train[0]).save('2-ConvBlock/cifar_index_1.png')
plt.imshow(x_train[0])
plt.show()


def model(data_train, data_test, y_train,y_test):
    # 数据归一化
    data_train = data_train / 255
    data_test = data_test / 255

    # 独热编码标签
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    """构建模型，创建一个简单的卷积神经网络模型。"""
    model = models.Sequential()

    # 第一个卷积层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # 32个卷积核，输出32个特征图，卷积核大小为3*3，激活函数为relu，由于步长默认为1，所以输出特征图大小为28-3+1=26，即26*26*32
    # 若图像为28*28*3，则卷积核大小为3*3*3，输出特征图大小为26*26*32
    model.add(layers.MaxPooling2D((2, 2)))
    # 执行最大池化操作，对于输入的每个特征图，它在一个小的（通常为2x2或3x3）窗口上滑动，并输出该窗口内的最大值。此处为2*2的窗口，步长默认为窗口大小，所以输出特征图大小为13*13*32

    # 第二个卷积层
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # 输入特征图大小为13*13*32，卷积核大小为3*3*32，输出特征图大小为13-3+1=11，即11*11*64
    model.add(layers.MaxPooling2D((2, 2)))
    # 输出特征图大小为5*5*64

    # 第三个卷积层
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # 输入特征图大小为5*5*64，卷积核大小为3*3*64，输出特征图大小为5-3+1=3，即3*3*64

    # 全连接层
    model.add(layers.Flatten())
    # 将3*3*64的特征图展平为一维向量，即576
    model.add(layers.Dense(64, activation='relu'))
    # 添加全连接层，输出64个神经元
    model.add(layers.Dense(10, activation='softmax'))
    # 添加全连接层，输出10个神经元，激活函数为softmax

    """编译模型 编译模型时需要指定优化器、损失函数和评估指标。"""
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 优化器为adam，损失函数为交叉熵损失，评估指标为准确率

    """训练模型 训练模型时需要指定训练数据、标签、批次大小和训练轮数。"""
    model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_split=0.1)

    """打印模型摘要，查看参数数量"""
    model.summary()

    """评估模型"""
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print(f'Test accuracy: {test_acc}')
