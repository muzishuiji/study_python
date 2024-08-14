class MulLayer:
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    # dout：上游传过来的导数
    def backward(self, dout):
        dx = dout * self.y # 翻转x和y
        dy = dout * self.x
        return dx, dy
    

apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward 
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer(apple_price, tax)
print(price) # 220

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num= mul_apple_layer.backward(dapple_price)
# 正向传播时输入变量的导数
price(dapple, dapple_num, dtax) # 2.2 110 200


