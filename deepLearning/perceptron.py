# 感知机
# 感知机的多个输入信号有各自固有的权重，这些权重发挥着控制各个信号的重要性的作用
# 权重越大，对应权重信号的重要性就越高。

# 相同构造的感知机，只需要适当的调整参数值，就可以像变色龙表演不同的角色一样，变身与门、非门、或门。
# 用python实现逻辑电路与门
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if(tmp < theta):
        return 0
    else:
        return 1
AND(0,0) # 0
AND(1,0) # 0
AND(0,1) # 0
AND(1,1) # 1

def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.3
    tmp = x1 * w1 + x2 * w2
    if(tmp < theta):
        return 0
    else:
        return 1
OR(0,0) # 0
OR(1,0) # 1
OR(0,1) # 1
OR(1,1) # 1

def XOR(x1, x2):
    w1, w2, theta = 0.5, -0.5, 0.3
    tmp = x1 * w1 + x2 * w2
    if(tmp < theta):
        return 0
    else:
        return 1
XOR(0,0) # 0
XOR(1,0) # 1
XOR(0,1) # 1
XOR(1,1) # 0

import numpy as np
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
# 感知机的计算
np.sum(w * x) + b

# 使用权重和偏置实现与门
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7
    tmp = np.sum(x * w) + b 
    if(tmp <= 0):
        return 0
    else:
        return 1

AND(0,0) # 0
AND(1,0) # 0

# w1,w2是输入信号重要性的参数，偏置b表示调整神经元被激活的容易程度

# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b 
    if(tmp <= 0):
        return 0
    else:
        return 1

NAND(0,0) # 1
NAND(1,0) # 1

# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b 
    if(tmp <= 0):
        return 0
    else:
        return 1

OR(0,0) # 0
OR(1,0) # 1
OR(0,1) # 1
OR(1,1) # 1

# 仅通过设置权重和偏置能实现不同的门电路
import numpy as np
def XOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, -0.5])
    b = -0.2
    tmp = np.sum(x * w) + b 
    if(tmp <= 0):
        return 0
    else:
        return 1

XOR(0,0) # 0
XOR(1,0) # 1
XOR(0,1) # 0
XOR(1,1) # 0

# 感知机的局限性就在于它只能表示一条直线分割的空间，而弯曲的曲线无法用感知机表示
# 由曲线分割的空间称为非线性空间，直线分割成的空间为线性空间
# 单层感知机无法表示异或门 或者 单层感知机无法分离非线性空间，我们可以通过组合感知机来实现异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
XOR(0,0) # 0
XOR(1,0) # 1
XOR(0,1) # 1
XOR(1,1) # 0

# 异或门是一种多层结构的神经网络。叠加了多层的感知机称为多层感知机
# 单层感知机无法表示的东西，通过增加一层就可以解决
# 通过叠加层，感知机能更加灵活的表示很多逻辑


# 通过与非门的组合就能实现计算机，而与非门可以使用感知机实现，那么通过组合感知机也可以表示计算机
# 理论上2层感知机就能构建计算机


# 总结
# 1. 感知机是具有输入和输出的算法。给点一个输入后，将输出一个既定的值
# 2. 感知机将权重和偏置设定为参数
# 3. 使用感知机可以表示与门和或门等逻辑电路
# 4. 异或门无法通过单层感知机来表示
# 5. 使用两层感知机可以表示异或门
# 6. 单层感知机只能表示线性空间，而多层感知机可以表示非线性空间
# 7. 多层感知机（理论上）可以表示计算机