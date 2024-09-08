class SGD:
    def __init__(self, lr=0.01):
        # 初始化学习率
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            # 用权重参数 - 梯度*学习率来更新权重参数
            params[key] -= self.lr * grads[key]

# 使用sgd类，可按如下方式进行神经网络的参数的更新
# network = TwoLayerNet(...)
# optimizer = SGD()
# for i in range(10000):
#     ...
#     x_batch, t_batch = get_mini_batch(...)
#     grads = network.gradient(x_batch, t_batch)
#     params = network.params
#     optimizer.update(params, grads)
#     ...

# 很多深度学习框架都实现了各种最优化方法，并且提供了可以简单切换这些方法的构造。比如lasagne深度学习框架
# 在updates.py 这个文件中以函数的形式集中实现了最优化方法，用户可以从中选择自己想使用的最优化方法

# 如果函数的形状非均向，比如呈延伸状，搜索的路径就会非常低效。
# 我们需要比单纯朝梯度方向前进的SGD更聪明的方法，SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。
# 基于SGD的最优化的更新路径：呈“之”字形朝最小值移动，效率低