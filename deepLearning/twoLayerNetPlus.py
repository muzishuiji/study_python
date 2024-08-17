# 通过误差反向传播计算关于权重参数的梯度
import sys, os

from deepLearning.layer.sofmaxWithLoss import SoftmaxWithLoss
sys.path.append(os.pardir)

import numpy as np
from common.layers import *
from commomn.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        # 初始化第1层权重矩阵
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # 初始化第2层权重矩阵
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        # 存神经网络各层的计算结果
        # 有序字典，可以记住向字典里添加元素的顺序
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    # x: 输入数据，t: 监督数据
    def loss(self, x, t):
        y = self.predict(x)
        # 交叉熵误差，计算最后一层的误差
        return self.lastLayer.forward(y, t)
    
    def accurancy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # x:输入数据，t: 监督数据
    # 基于数值微分计算各权重参数的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) 
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) 
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) 
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) 
        return grads
    
    # 基于误差反向传播计算各权重参数的梯度
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        [grads['W1'], grads['b1']] = [self.layers['Affine1'].dW, self.layers['Affine1'].db]
        [grads['W2'], grads['b2']] = [self.layers['Affine2'].dW, self.layers['Affine2'].db]
        return grads
    
# 像这样通过将神经网络的组成元素以层的方式实现，可以轻松的构建神经网络
# 用层进行模块化具有很大优点，因为想另外构建一个神经网络，
# 比如5层，10层，20层的神经网络，只需要像组装乐高积木那样添加必要的层就可以了
# 通过各个层内部实现的正向传播和反向传播，就可以正确进行时被处理或学习所需的梯度

# 数值微分的优点是实现简单，一般情况下不太容易出错，而误差反向传播法的实现复杂，容易出错。
# 经常会比较数值微分的结果和误差反向传播法的结果，以确认误差反向传播法的结果是否正确。
# 确认微分求出的梯度结果和误差反向传播法求出的结果是否一致（是否非常相近）的操作称为梯度确认


