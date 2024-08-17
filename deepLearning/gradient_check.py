import sys, os

from deepLearning.twoLayerNetPlus import TwoLayerNet
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# 抽样选取数据
x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

