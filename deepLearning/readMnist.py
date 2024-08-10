# 读入minist数据
import sys,os

from deepLearning import sigmoid, softmax

sys.path.append(os.pardir) # 为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist

# 第一次调用会花费几分钟
# (训练图像、训练标签)，（测试图像，测试标签）
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 输出各个数据的形状
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, )
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000, )

# python有pickle这个功能，可以将程序运行中的对象保存为文件。如果加载保存过的pickle文件，可以立刻复原之前程序运行中的对象
# 显示mninst图像
import sys, os
import pickle
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

sys.path.append(os.pardir)

def img_show(img):
    # 将numpy数组的图像数据转换成pil用的数据对象
    pil_img = Image.fromarray(np.unit8(img))
    pil_img.show()

(x_train, t_train) , (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)


# 用这三个函数实现神经网络的推理处理，然后，评价它的识别精度，即能在多大程度上争取分类

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# 借助上面定义的函数实现神经网络的推理处理
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy: " + str(float(accuracy_cnt) / len(x)))

# load_mnist函数normalize设置为true，函数内部会进行转换，将图像的各个像素除以255，使得数据的值在0.0-1.0的范围内
# 像这样把数据限定到某个范围内的处理称为正规化（normalization）
# 对神经网络的输入数据进行某种既定的转换称为预处理（pre-processing）
# 很多预处理都会考虑数据的整体分布，数据预处理的有效性在提高识别性能和学习的效率等众多实验中得到证明
# 输出神经网络的各层的权重的形状
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
print(x.shape)
# (784, )
x[0].shape
# (784, )
print(W1.shape)
# (784, 50)
print(W2.shape)
# (50, 100)
print(W3.shape)
# (100, 10)

# 批处理对计算机的运算大有利处，可以大幅缩短每张图像的处理时间。批处理一次性激素颜大型数组要比分开逐步计算哥哥小型数组速度要快

# 加入批处理
x, t = get_data()
network = init_network()
batch_size = 100 # 批处理的数量
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    # axis=1沿着第1维方向（矩阵的第0维是列方向，第1维是行方向）找到值最大的元素的索引
    p = np.argmax(y_batch, axis=1) # 获取概率最高的元素的索引
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print("Accuracy: " + str(float(accuracy_cnt) / len(x)))


# 找最大元素的索引的例子
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=1)
print(y)
# [1 2 1 0]

# 将分类的结果和实际的答案做比较，观测推理的准确度
y = np.array([1,2,1,0])
t = np.array([1,2,0,0])
print(y==t)
# [ True  True False  True]
print(np.sum(y==t))
# 3


# 神经网络章节的小结
# - 神经网络中的激活函数使用平滑变化的sigmoid函数或relu函数
# - 巧妙的使用numpy多维数组，可以高效的实现神经网络
# - 机器学习的根本问题大体上可以分为回归问题和分类问题
# - 关于输出层的激活函数，回归问题中一般用恒等函数，分类问题中一般用softmax函数
# - 分类问题中，输出层的神经元数量设置为要分类的类别数
# - 输入数据的集合称为批。以批为单位进行推理处理，能实现高速的运算