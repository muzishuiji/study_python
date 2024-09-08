import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100) # 1000个数据
node_num = 100 # 各隐藏层的节点数
hidden_layer_size = 5 # 隐藏层有5层
activations = {} # 保存所有激活值的字典

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
    # 前一层的节点数越多，要设定的目标节点的初始值的权重尺度越小
    # 权重值的调整，使得后面的图像变得越歪斜，呈现了比之前更有广度的分布
    # 因为各层间传递的数据有适当的广度，所以sigmoid函数的表现力不受限制，有望进行高效的学习
    w = np.random.randn(node_num, node_num) * 1 / np.sqrt(node_num)
    # 上一层的激活值加入计算
    z = np.dot(x, w)
    # 
    a = sigmoid(z) # sigmoid函数
    activations[i] = a

# 我们使用标准差为1的高斯分布作为权重初始值计算得到各层激活值activations
# 将保存在activations中的各层数据画成直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + '-layer')
    plt.hist(a.flatten(), 30, range(0, 1))
plt.show()


# 合适的权重尺度，如果前一层的节点数为N，则初始值使用标准差为 1/sqrt(N) 的高斯分布进行初始化

