import numpy as np
from dataset.minist import load_minist
from twoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_minist(normalize=True, one_hot_label=True)

train_loss_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # 根据计算出的梯度更新参数，使得整体的效果朝着损失函数的值越来越小的方向发展
    # 学习的过程就是不断优化的过程
    for key in ('w1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程，将每次学习的损失函数的值记录下来，就是交叉熵误差的结果
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

# epoch是一个单位，一个epoch表示学习中所有训练数据均被使用过一次的更新次数
# 对于10000笔训练数据，用大小100笔数据的mini-batch进行学习时，重复随机梯度下降法100次，所有数据都被看过了
# 此时，100次就是一个epoch

import numpy as np
from dataset.minist import load_minist
from twoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_minist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)    

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # 根据计算出的梯度更新参数，使得整体的效果朝着损失函数的值越来越小的方向发展
    # 学习的过程就是不断优化的过程
    for key in ('w1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程，将每次学习的损失函数的值记录下来，就是交叉熵误差的结果
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# 通过观察训练数据的识别精度和测试数据的识别精度，
# 可以观测是否随着学习的进行，训练数据和测试数据评价的识别精度都提高了，以评估这次的学习是否发生过拟合的现象。

# 神经网络学习的目标：找到尽可能小的损失函数值。