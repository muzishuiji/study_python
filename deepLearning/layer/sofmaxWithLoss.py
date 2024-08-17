from deepLearning import softmax
from deepLearning.training import cross_entropy_error


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None # softmax的输出
        self.t = None # 监督数据（one-ht vector）

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        # 计算推理和测试数据的差分
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    # 输出的概率论值为1，进行反向传播
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    
# 反向传播时，将要传播的值除以批的大小，传递给前面的层的时单个数据的误差
# 梯度也是深度学习的推理结果和测试数据之间的差值，在优化过程中，会将权重参数沿梯度方向进行微小的更新