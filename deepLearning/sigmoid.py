# 最简单的阶跃函数
def step_function(x):
    if x >0:
        return 1
    else:
        return 0
    

# 实现可以接收numpy数组的阶跃函数
import numpy as np
def step_function(x):
    y = x > 0
    # 转换numpy数组的类型，参数指定期望的类型
    return y.astype(int) 

step_function(np.array([-1.0, 2.0]))

# 阶跃函数的图形
# 图像看起来很像到达一定阶梯后跃升，呈阶梯式变化
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1,1) # 指定y轴的范围
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# sigmoid 能处理numpy数组，根据numpy的广播功能，标量和numpy数组之间的运算可以顺利进行
# sigmoid函数的图形
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
# sigmoid 对神经网络的学习具有重要意义
# 感知机中神经元之间流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号

# 线形函数与非线形函数
# - 线性函数：输出值是输入值的常数倍的函数称为线性函数，是一条笔直的直线
# - 非线性函数：非呈现一条直线的函数

# 使用线形函数时，无法发挥多层网络带来的优势，为了发挥叠加层带来的优势，激活函数必须使用非线形函数


