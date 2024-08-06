import numpy as np
# 神经网络的早期使用sigmoid函数，后期使用relu函数
def relu(x):
    return np.maximum(0, x)

