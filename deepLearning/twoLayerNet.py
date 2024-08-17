# 手写数字识别的神经网络，以2层神经网络为对象
import sys, os

from deepLearning import sigmoid, softmax
from deepLearning.training import cross_entropy_error
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] =  weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    # x: 输入数据，t: 监督数据
    def loss(self, x, t):
        y = self.predict(x)
        # 交叉熵误差
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # x: 输入数据，t: 监督数据, 计算每个权重和偏置的梯度
    # 基于数值微分计算参数的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        # 第1层权重的梯度
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 第2层权重的梯度
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 第1层偏置的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 第2层偏置的梯度
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
net.params['W1'].shape # (784, 100)
net.params['b1'].shape # (100,)
net.params['W2'].shape # (100, 10)
net.params['b2'].shape # (10,)

x = np.random.rand(100, 784) # 伪输入数据（100笔）
y = net.predict(x)

x = np.random.rand(100, 784) # 伪输入数据
t = np.random.rand(100, 10) # 伪正确解标签
grads = net.numerical_gradient(x, t) # 计算梯度
net.params['W1'].shape # (784, 100)
net.params['b1'].shape # (100,)
net.params['W2'].shape # (100, 10)
net.params['b2'].shape # (10,)

