import numpy as np

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flag=True):
        if train_flag:
            # 随即生成和x形状相同的数组，将值比dropout_ratio打的元素设为True，值比dropout_ratio小的元素设置为False（要删除的神经元）
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
    


