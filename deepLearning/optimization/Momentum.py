import numpy as np
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # 输入val数组，返回一个和val数组形状相同，元素都为0的数组
                self.v[key] = np.zeros_like(val)

            for key in params.keys():
                # 在SGD的基础上增加了动量项这一影响因素
                # self.v[key] 上一次更新时物体的运动速度
                # 这个动量在交互地受到正方向和反方向的力会相互抵消，
                # 可以一定程度上减弱运动过程中的震荡，从而更快的朝着x轴方向靠近，减弱“之”字形的变动程度
                self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
                params[key] += self.v[key]
