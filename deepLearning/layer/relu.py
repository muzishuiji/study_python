# 激活层的relu层，默认输入时numpy数组

class Relu:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        # 复制一份x，将x<=0的值置为0
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
import numpy as np
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)
# [[ 1.  -0.5]
#  [-2.   3. ]]
# 数组中满足表达式的被置为True，否则为False
mask = (x <= 0)
print(mask)
# [[False  True]
#  [ True False]]




