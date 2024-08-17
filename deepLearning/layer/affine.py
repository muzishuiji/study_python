# 
import numpy as np
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.db = None
        self.dW = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) = self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    


# softmax 会将函数的输入值正规化后再输出。（将输出值的和调整为1）之后再输出
# 神经网络的反向传播会把这个差分表示的误差传递给前面的层，这是神经网络学习中的重要性质
