from deepLearning.layer.addLayer import AddLayer
from deepLearning.layer.mulLayer import MulLayer


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orang_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orang_price)
price = mul_tax_layer.forward(all_price, tax)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorang_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorang_price)


print(price) # 715
print(dapple_num, dapple, dorange, dorange_num, dtax) # 110 2.2 135 3.3 650

# 计算图中层的实现非常简单，使用这些层可以进行复杂的导数计算