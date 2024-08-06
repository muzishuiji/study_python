import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
# [1, 2, 3, 4]
np.ndim(A)
# 1
A.shape
# (4,)
A.shape[0]
# 4

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
np.ndim(B)
# 2
B.shape
# (3, 2)

# 二维数组也称为矩阵
# 多维数组的矩阵计算，需要保证矩阵A的列数（第1维的元素个数）和矩阵B的第0维的元素个数（行数）相等，否则会出错
# 运算结果矩阵C的形状是矩阵A的行数和矩阵B的列数构成的
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)
# (2,3)
print(B.shape)
# (3,2)
np.dot(A, B)
# array([[22, 28],
#        [49, 64]])


# 通过矩阵的乘积进行神经网络的运算
# 表达式
# 1. x1 * 1 + 2 * x2 = y1
# 2. x1 * 3 + 4 * x2 = y2
# 3. x1 * 5 + 6 * x2 = y3

X = np.array([1, 2])
X.shape
# (2,)
W = np.array([[1, 3, 5], [2, 4, 6]])
W.shape
# (2,3)
Y = np.dot(X, W)
Y
# array([ 5, 11, 17])

# 使用多维数组来实现，两个信号x1, x2，权重分别为w1, w2，偏置为b，计算输出的表现式

# 实现第0层到第1层的信号传递
from deepLearning import sigmoid
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print(W1.shape)
# (2, 3)
print(X.shape)
# (2,)
print(B1.shape)
# (3,)
A1 = np.dot(X, W1) + B1
# array([0.3, 0.7, 1.1])

# 从a到z，使用激活函数h()将信号a转换成信号z，我们使用sigmoid函数作为激活函数
Z1 = sigmoid(A1)
print(A1)
# array([0.3, 0.7, 1.1])
print(Z1)
# array([0.57444252, 0.66818777, 0.75026011])

# 实现第1层到第2层的信号传递
W2 = np.array([0.1, 0.4], [0,2, 0.5], [0.3, 0.6])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)
# array([0.51615984, 0.59514535])

# 实现第2层到输出层的信号传递
W3 = np.array([0.1, 0.3], [0,2, 0.4])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
# 输出层的激活函数和隐藏层的有所不同，输出层的激活函数用σ()（读sigma）表示
def idetity_function(x): # 恒等函数，即输入和输出一致
    return x
Y = idetity_function(A3) # 或者Y=A3
print(Y)
# array([0.31682708, 0.69627909])

# 输出层的激活函数，要根据求解问题的性质决定
# 1. 回归问题：输出层使用恒等函数
# 2. 二元分类问题可以使用sigmoid函数
# 3. 多元分类问题可以使用softmax函数


# 最终代码，实现输入到输出的前向处理
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

# 封装输入到输出的前向处理
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = idetity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
# array([0.31682708, 0.69627909])
