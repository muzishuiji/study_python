# 导数是某个瞬间的变化量，即表示微小变化的h无限趋近0

# 代码求函数的导数
def numerical_diff(f,x):
    h = 10e-50
    return (f(x+h) - f(x)) / h

# numerical_diff 的名称来源于数值微分的英文 numerical differentiation
# 数值微分就是用数值方法近似求解函数的导数的过程

# 上述代码的问题，使用过小的微小值会导致舍入误差，所谓舍入误差
# 是指因省略小数的精细部分的数值，比如，小数点第8位以后的数值，而造成的最终计算结果上的误差
import numpy as np
np.float32(1e-50)
# 0.0

# 数值微分含有误差，为减小误差，可以计算f在x+h和x-h之间的差分
# 这种计算方式以x为中心，计算它左右两边的差分，也称为中心差分
# （而x+h和x之间的差分称为前向差分，x和x-h之间的差分称为向后差分）

def numerical_diff(f,x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 基于数学式的推导求导数的过程，用“解析性”一词，称为“解析性求解”或“解析性求导”
# 解析性求导得到的导数是不含误差的“真的导数”

# 定义一个简单函数
def function_1(x):
    return 0.01*x**2 + 0.1*x

# 绘制函数图像
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1) # 以0.1为单位。从0-20的数组x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y) # 绘制x和y的图像
plt.show()

numerical_diff(function_1, 5)
# 0.1999999999990898
numerical_diff(function_1, 10)
# 0.2999999999986347

# 真的导数：df(x)/dx = 0.02x + 0.1,x=5,x=10分别为0.2和0。3
# 数值微分和解析性求导虽然严格意义上并不一致，但误差非常小，基本可以认为他们是相等的
# 数值微分的值作为斜率画一条直线，这些直线确实对应函数的切线


# 偏导数:有多个变量的函数的导数称为偏导数
# 计算平方和的简单函数 f(x0,x1) = x0^2 + x1^2
def function_2(x):
    return x[0]**2 + x[1]**2
    # 或者return np.sum(x**2)

# 偏导数需要将多个变量中的某个变量定位目标变量，并将其他变量固定为某个值，将目标变量以外的值固定到某些特定的值上
# 就定义了新函数，对新函数应用数值微分，得到偏导数
# 1. x0=3,x1=4,求关于x0的偏导数
def function_tmp1(x0):
     return x0*x0 + 4.0**2.0

numerical_diff(function_tmp1, 3.0)
# 6.00000000000378

# 梯度
# 由全部向量汇总而成的向量称为梯度
# 将多个变量产生的数值微分的结果放到一个数组里，代码实现梯度
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x) # 生成和x形状相同的数组
    for idx in range(x.size):
        temp_val = x[idx]
        x[idx] = temp_val + h
        fxh1 = f(x)

        x[idx] = temp_val - h
        fxh2 = f(x)

        x[idx] = temp_val
        grad[idx] = (fxh1 - fxh2) / (2*h)

    return grad

def function_2(x):
    return x[0]**2 + x[1]**2
# 分别计算下（3，4）、（0，2）、（3，0）处的梯度
numerical_gradient(function_2, np.array([3.0, 4.0]))
# array([6., 8.])
numerical_gradient(function_2, np.array([0.0, 2.0]))
# array([0., 4.])
numerical_gradient(function_2, np.array([3.0, 0.0]))
# array([6., 0.])

# 负梯度方向是梯度法中变量的更新方向
# 将f(x0, x1)的梯度画出来后，发现像指南针一样，所有的箭头都指向透一点，其次，发现离最低处越远，箭头越大
# 梯度会指向各点出的函数值降低的方向。更严格的讲，梯度指示的方向是各点处的函数值减小最多的方向。
# 方向导数=cos(θ)x梯度（θ是方向导数的方向与梯度方向的夹角），因此，所有的下降方向中，梯度方向下降最多


# 梯度表示的是各点处的函数值减小的最多的方向，因此，无法保证梯度所指的方向就是函数的最小值或者真正应该前进的方向
# 实际上，在复杂的函数中，梯度指示的方向基本上都不是函数值最小处。

# 函数的极小值、最小值以及被称为鞍点（saddle point）的地方，梯度为0
# 极小值是局部最小值，也就是限定在某个范围内的最小值，
# 鞍点从某个方向上看是极大值，从另一个方向上看是极小值
# 虽然梯度法是要寻找梯度为0的地方，但是那个地方不一定就是最小值（也有可能是极小值或鞍点）
# 当函数很复杂且呈扁平状时，学习可能会进入一个几乎平坦的地区，陷入被称为“学习高原”的无法前进的停滞期

# 根据目的是寻找最大值还是最小值，梯度法的叫法有所不同，严格的讲，寻找最小值的梯度法称为梯度下降法，寻找最大值的梯度法称为梯度上升法
# 但通过反转损失函数的符号，求最细哦啊值的问题和球最大值的问题都会变成相同的问题，因此上升还是下降本质上并不重要
# 一般来说，神经网络（深度学习）中，梯度法主要是指梯度下降法
# 用数学式表示梯度法，这个式子会反复执行，逐渐缩小函数值
#   x0 = x0 - ηθf/θx0
#   x1 = x1 - ηθf/θx1
# η表示更新量，在神经网络的学习中，称为学习率，学习率决定在一次学习中，应该学习多少，以及在多大程度上更新参数

# python来实现梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        # x减去每次梯度计算出来的结果乘以学习率
        x -= lr * grad
    return x

# 用梯度法求f(x0+x1) = x0^2+x1^2的最小值
def function_2(x):
    return x[0]**2 + x[1]**2
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# array([-6.11110793e-10,  8.14814391e-10]) 
# 计算结果非常接近（0，0），真的最小值就是（0，0）
# 用图来表示梯度法的更新过程，会发现原点是最低的地方，函数的取值一步步向原点靠近

# 学习率过大的例子：lr=10.0
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
# array([ -2.58983747e+13,  -1.29524862e+12])

# 学习率过小的例子：lr=1e-10
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
# array([-2.99999994,  3.99999992])

# 实验结果表明：
# 学习率过大的话，会发散成一个很大的值，学习率过小的话，基本上没怎么更新就结束了，设置合适的学习率是一个很重要的问题

# 像学习率这样的参数称为超参数，这是一种和神经网络的参数（权重和偏置）性质不同的参数。
# 相对于神经网络的权重参数是通过训练数据和学习算法自动获得的，学习率这样的超参数则是人工设定的
# 一般来说，超参数需要尝试多个值，以便找到一种可以使学习顺利进行的设定

# 以一个简单的神经网络为例，实现求梯度的代码
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2,3) # 用高斯分布初始化权重

    def predict(self, x):
        return np.dot(x, self.W) # 预测
    
    # 求损失函数值
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
    
# 试着用一下
net = simpleNet()
print(net.W) # 权重参数
# [[ 0.47355232   0.9977393    0.84668094],
#  [ 0.85557411   0.03563661   0.69422093]])

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
# [ 1.05414809  0.63071653  1.1328074]
np.argmax(p) # 最大值的索引
# 2

t = np.array([0, 0, 1]) # 正确解标签
net.loss(x, t)
# 0.92806853663411326

# 求梯度
def f(W):
     return net.loss(x, t)

# 实际运行的numerical_gradient 和定义的稍有改动，为了对应多维数组，具体要查看源码
dW = numerical_gradient(f, net.W)
print(dW)
# [[ 0.21924763  0.14356247 -0.36281009]
#  [ 0.32887144  0.2153437  -0.54421514]]

# 可以看到如果w11的值增加h，损失函数的值就会增加0.2h
# 如果为w23增加h，损失函数的值将减小0.5h，w23应向正方向更新，w11应向负方向更新，至于更新的程度，w23比w11的贡献要打

# python中定义的是简单的函数，可以使用lambda表示法
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
