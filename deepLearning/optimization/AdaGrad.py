import numpy as np
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

            for key in params.keys():
                # 保存了所有梯度值的平方和
                self.h[key] += grads[key] * grads[key]
                # 1e7用来防止0做除数的情况
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e7)

# 经过上述处理后，函数的取值高效地向着最小值移动，由于y轴方向上的梯度较大，因此刚开始变动较大
# 但后面会根据这个较大的变动按比例进行调整，奸笑更新的步伐，因此，y轴方向上的更新程度减弱，之字形的变动程度减弱
