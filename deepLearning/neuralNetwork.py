# 神经网络的重要形式是可以自动的从数据中学习合适的权重参数
#摆脱人工决定参数的固有限制，从而可以快速的表示非常多复杂的函数

# 激活函数的作用在于决定如何激活输入信号的总和
# 激活函数是链接感知机和神经网络的桥梁


a = b + w1 * x1 + w2 * x2
#  h(x) 函数会讲输入信号的总和转换为输出信号，这种函数一般称为激活函数
y = h(a) 
# 一般而言：“朴素感知机”是指单层网络，指的是激活函数使用了阶跃函数（阶跃函数是指一旦超过阈值，就切换输出的函数）的模型。
# “多层感知机”是指神经网络，即使用sigmoid函数等平滑的激活函数的多层网络
# 将激活函数从阶跃函数换成其他函数，就可以进入神经网络的世界了