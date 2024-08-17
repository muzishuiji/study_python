# python实现sigmoid层
import numpy as np
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        # 正向传播时将输出爆存在了实例变量out中
        self.out = out
        return out
    
    def backward(self, dout):
        # 反向传播时，使用变量out进行计算
        dx = dout * (1.0 - self.out) * self.out
        return dx

# 
Y = np.dot(X, W) + B