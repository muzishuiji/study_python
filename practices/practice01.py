# round: 浮点数的四舍五入
round(1.23, 1)
# 浮点数一个普遍的问题是他们并不能精确的表示十进制数，如果你想要更精确，可以使用decimal模块

from decimal import Decimal
# decimal模块的一个主要特征是允许你控制计算的每一方面，包括数字位数和四舍五入运算
# 为了这样做，你得先创建一个本地上下文并更改它的设置
from decimal import localcontext
import math

import pytz
a = Decimal('1.3')
b = Decimal('1.7')
print(a / b)
# 0.7647058823529411764705882353
with localcontext() as ctx:
    ctx.prec = 3
    print(a / b)
# 0.765
    
with localcontext() as ctx:
    ctx.prec = 50
    print(a / b)
# 0.76470588235294117647058823529411764705882352941176
# 大多数应用领域一点点误差会被允许的，而且原生的浮点数运算也要快的多（大量运算的时候速度是非常重要的）
# 但要视情况而定，有些误差可以通过更精确的计算方法解决
# decimal同行吃那个用于涉及金融的领域，这类领域对精确度要求比较高，不允许误差
# 一个非常方便的格式化数据的方法format
x = 1234.56789
format(x, '0.2f') 
# '1234.57'
format(x, '>10.1f')
# '    1234.6'
format(x, '^10.1f')
# '  1234.6  '
format(x, '<10.1f')
# '1234.6    '
format(x, ',')
# '1,234.56789'

format(x, '0,.1f')
# '1,234.6'

# 使用指数标记法
format(x, 'e')
# '1.234568e+03'
format(x, '0.2E')
# '1.23E+03'

# 同时指定宽度和精度的一般形式 '[<>^]?width[,]?(.digits)?', ?代表可选部分
format(-x, '0.1f')
# '-1234.6'
# 可以使用translate对数字中的字符做映射
format(x, ',').translate({ ord('.'): ',', ord(','): '.'})

# python中很常见用%来格式化数字，但format能力更强更灵活
'%0.2f' % x
# '1234.57'

'%10.1f' % x
# '    1234.6'
'%1-0.1f' % x
# '1234.6    '

x = 1234
bin(x)
# '0b10011010010' 二进制
oct(x)
# '0o2322' 八进制
hex(x)
# '0x4d2' 十六进制

# format可以输出不带前缀的数据
format(x, 'b')
# '10011010010'
format(x, 'o')
# '2322'
format(x, 'x')
# '4d2'

y  = -1234
format(y, 'x')
# '-4d2'

# 如果你想产生一个无符号值，你需要增加一个指示最大长度的值
format(2 ** 32 + y, 'b')
# '11111111111111111111101100101110'
format(2 ** 32 + y, 'x')
# 'fffffb2e'

# 其他进制数据转换成十进制整数，使用int函数即可
int('4d2', 16)
# 1234
int('10011010010', 2)
# 1234

# 在python中八进制的前缀是0o而不是0
import os
os.chmod('script.py', 0o755) # not 0755

# 字节字符串--> 整数
data = b'\x00\x124V\x00x\x90\xab\x00\xcd\xef\x01\x00#\x004'
len(data) # 16
int.from_bytes(data, 'little')
# 69120565665751139577663547927094891008
int.from_bytes(data, 'big')
# 94522842520747284487117727783387188
# 整数 --> 字节字符串
x = 94522842520747284487117727783387188
x.to_bytes(16, 'big')
# b'\x00\x124V\x00x\x90\xab\x00\xcd\xef\x01\x00#\x004'
x.to_bytes(16, 'little')
# 大整数和字符串之间的转换不常见，在密码学或者网络中会出现
# ipv6网络地址使用128位的整数表示
# little和big指定了构建整数的字节的低位高位排列方式
x = 0x01020304
x.to_bytes(4, 'big')
# b'\x01\x02\x03\x04'
x.to_bytes(4, 'little')
# b'\x04\x03\x02\x01'

# 将一个整数转换为字节字符串，需要先使用int.bit_length() 方法来获取需要多少字节位来存储这个值
x = 523 ** 23
# 335381300113661875107536852714019056160355655333978849017944067
x.bit_length()
# 208
nbytes, rem = divmod(x.bit_length(), 8)
if rem:
    nbytes += 1
x.to_bytes(nbytes, 'little')
# b'\x03X\xf1\x82iT\x96\xac\xc7c\x16\xf3\xb9\xcf...\xd0'

# 可以使用complex和字面量声明的形式来生成复数
a = complex(2, 4)
# (z+4j)
b = 3 - 5j
# (3-5j)
# 实部
a.real
# 2.0
# 虚部
a.imag
# 4.0
# 共轭复数
a.conjugate()
# (2-4j)

# 复数的数学运算
a + b
# (5-1j)
a - b
# (26+2j)
a * b
# (-0.4117647058823529+0.6470588235294118j)
a / b
# 4.47213595499958

# 要获取复数的正弦，余弦，或平方根，使用cmath模块
import cmath
cmath.sin(a)
# (24.83130584894638-11.356612711218174j)
cmath.cos(a)
# (-11.36423470640106-24.814651485634187j)
cmath.exp(a)
# (-4.829809383269385-5.5920560936409816j)

# numpy可以构造一个复数数组，并在这个数组上执行各种操作
import numpy as np
a = np.array([2+3j, 4+5j, 6-7j, 8+9j])
# array([ 2.+3.j, 4.+5.j, 6.-7.j, 8.+9.j])
a + 2
# array([ 4.+3.j, 6.+5.j, 8.-7.j, 10.+9.j])
np.sin(a)
# array([ 9.15449915 -4.16890696j, -56.16227422 -48.50245524j, -153.20827755-526.47684926j, 4008.42651446-589.49948373j])

# 标准数学函数math不能处理复数，要使用cmath模块
import cmath
cmath.sqrt(-1)
# 1j

# 创建正无穷、负无穷，NaN
a = float('inf')
b = float('-inf')
c = float('nan')

# 测试值的存在
math.isinf(a)
# True
math.isnan(c)
# True

# nan的所有运算结果都是nan
c + 23
# nan
c / 2
# nan
math.sqrt(c)
# nan

# nan和自己比较会返回false
d = float('nan')
c == d
# False
c is d
# False

# 对正无穷大和负无穷大做操作可能返回正无穷或者负无穷或nan
# fractions模块可以被用来执行包含分数的数学运算
from fractions import Fraction
a = Fraction(5, 4)
b = Fraction(7, 16)
print(a + b)
# 27/16
print(a + b)
# 35/64

c = a * b
c.numerator
# 35
c.denominator
# 64
float(c)
# 0.546875
# Limiting the denominator of a value
print(c.limit_denominator(8))
# 4/7
# Converting a float to a fraction
x = 3.75
y = Fraction(*x.as_integer_ratio())
y
# Fraction(15, 4)

# 在大多数程序中一般不会出现分数的计算问题，但是有时候还是需要用到的
# 在一个允许接受分数形式的测试单位并以分数形式执行运算的程序中，直接使用分数可以减少手动转换为小数或浮点数的工作


# 底层实现中，numpy数组使用了c或者fortran语言的机制分配内存，也就是
# 它们是一个非常大的连续的并且由同类型数据组成的内存区域
# 你可以很轻松的构造一个10000*10000的浮点数二维网格

grid = np.zeros(shape=(10000, 10000),dtype=float)
grid
# array([[ 0., 0., 0., ..., 0., 0., 0.],
# [ 0., 0., 0., ..., 0., 0., 0.],
# [ 0., 0., 0., ..., 0., 0., 0.],
# ...,
# [ 0., 0., 0., ..., 0., 0., 0.],
# [ 0., 0., 0., ..., 0., 0., 0.],
# [ 0., 0., 0., ..., 0., 0., 0.]])

# 所有普通操作都会作用在所有元素上
np.sin(grid)
# array([[-0.54402111, -0.54402111, -0.54402111, ..., -0.54402111,
#         -0.54402111, -0.54402111],
#     [-0.54402111, -0.54402111, -0.54402111, ..., -0.54402111,
#         -0.54402111, -0.54402111],
#     [-0.54402111, -0.54402111, -0.54402111, ..., -0.54402111,
#         -0.54402111, -0.54402111],
#     ...,
#     [-0.54402111, -0.54402111, -0.54402111, ..., -0.54402111,
#         -0.54402111, -0.54402111],
#     [-0.54402111, -0.54402111, -0.54402111, ..., -0.54402111,
#         -0.54402111, -0.54402111],
#     [-0.54402111, -0.54402111, -0.54402111, ..., -0.54402111,
#         -0.54402111, -0.54402111]])
import numpy as np
ax = np.array([1, 2, 3, 4])
ay = np.array([5, 6, 7, 8])
# 对数组的每一项进行多项式运算
ax * 2
# array([2, 4, 6, 8])
ax + 10
# array([11, 12, 13, 14])
ax + ay
# array([ 6, 8, 10, 12])
ax * ay
# array([ 5, 12, 21, 32])
def f(x):
    return 3 * x ** 2 - 2 * x + 7
f(ax)
# array([ 8, 15, 28, 47])
# numpy包含了很多math函数的替代，可以很方便的处理大型数组运算
# numpy有一个矩阵对象可以处理矩阵乘法、寻找行列式、求解线性方程等
import numpy as np
m = np.matrix([[1,-2,3], [0,4,5], [7,8,-9]])
# matrix([[ 1, -2,  3],
#         [ 0,  4,  5],
#         [ 7,  8, -9]])
# Return inverse
m.T
# matrix([[ 1, 0, 7],
#         [-2, 4, 8],
#         [ 3, 5, -9]])
m.I
# matrix([[ 0.33043478, -0.02608696, 0.09565217],
#         [-0.15217391, 0.13043478, 0.02173913],
#         [ 0.12173913, 0.09565217, -0.0173913 ]])
v = np.matrix([[2],[3],[4]])
# matrix([[2],
#         [3],
#         [4]])
m * v
# matrix([[ 8],
#         [32],
#         [ 2]])

# 你可以在numpy.linalg 中找到更多的操作函数
import numpy.linalg
# Determinant
numpy.linalg.det(m)
# 229.99999999999983
# Eigenvalues
numpy.linalg.eigvals(m)
# array([-13.11474312, 2.75956154, 6.35518158])
# Solve for x in mx = v
x = numpy.linalg.solve(m, v)
x
# matrix([[ 0.96521739],
#         [ 0.17391304],
#         [ 0.46086957]])
m * x
# matrix([[ 2.],
#         [ 3.],
#         [ 4.]])
v
# matrix([[2],
#         [3],
#         [4]])
# 如果你需要操作数组或者向量的话，numpy是一个不错的入口点

# random 模块有大量的函数来产生随机数和随机选择元素
import random
values = [1,2,3,4,5,6]
random.choice(values) # 2
# 随机提取n个元素
random.sample(values, 2) # [6, 2]
# 打乱序列中元素的顺序，会修改原数组
random.shuffle(values) 
values
# [2, 4, 6, 5, 3, 1]

# 生成随机整数 randint(0-10之间)
random.randint(0, 10) # 2

# 生成0-1范围内随机分布的浮点数吧
random.random()
# 0.9406677561675867
random.random()
# 0.133129581343897
random.random()
# 0.4144991136919316
# 获取N位随机数
random.getrandbits(200)
# 335837000776573622800628485064121869519521710558559406913275

# random模块是使用Mersenne Twister 算法来计算生成随机数，这是一个确定性算法，但是可以通过random.seed来修改初始化种子
random.seed()
random.seed(12345)
random.seed(b'bytedata')
# random模块还支持基于均匀分布，搞死分布和其他分布的随机生成函数
# random.uniform()计算均匀分布随机数， random.gauss()计算正态分布随机数
# 用ssl模块中相应的函数。 比如， ssl.RAND_bytes() 可以用来生成一个安全的随机字节序列。

# 日期转换逻辑
from datetime import timedelta
a = timedelta(days=2, hours=6)
b = timedelta(hours=4.5)
c = a + b
c.days
# 2
c.seconds
# 37800
c.seconds / 3600
# 10.5
c.total_seconds() / 3600
# 58.5

# 具体的时间和日期运损啊
from datetime import datetime
a = datetime(2012, 9, 23)
print(a + timedelta(days=10))
# 2012-10-03 00:00:00
b = datetime(2012, 12, 21)
d = b - a
d.days
# 89
now = datetime.today()
print(now)
# 2012-12-21 14:54:43.094063
print(now + timedelta(minutes=10))
# 2012-12-21 15:04:43.094063

# 在计算的时候，datetime会自动处理闰年
a = datetime(2012, 3, 1)
b = datetime(2012, 2, 28)
a - b
# datetime.timedelta(2)
(a - b).days()
# 2
c = datetime(2013, 3, 1)
d = datetime(2013, 2, 28)
(c - d).days()
# 1

# 大部分的日期和时间处理问题，datetime模块就够了
# 如果需要执行更加复杂的日期操作，比如处理时区，模糊时间范围，节假日计算等等，考虑使用dateutil模块
from dateutil.relativedelta import relativedelta
a = datetime(2012, 9, 23)
a + relativedelta(months=+1)
# datetime.datetime(2012, 10, 23, 0, 0)
a + relativedelta(months=+4)
# datetime.datetime(2013, 1, 23, 0, 0)
# Time between two dates
b = datetime(2012, 12, 21)
d = b - a
d
# datetime.timedelta(89)
d = relativedelta(b, a)
relativedelta(months=+2, days=+28)
d.months
# 2
d.days
# 28

# IBM：云服务、云计算、数据分析、安全服务、区块链
# 语言的lsp（语言服务协议）自动补全

# 计算最后的周五
from datetime import datetime, timedelta

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday']

def get_previous_byday(day_name, start_date=None):
    if start_date is None:
        start_date = datetime.today()
    day_num = start_date.weekday()
    day_num_target = weekdays.index(day_name)
    days_ago = (day_num - day_num_target) % 7
    if(days_ago == 0):
        days_ago = 7
    target_date = start_date - timedelta(days=days_ago)
    return target_date

get_previous_byday('Friday', datetime(2012, 12, 21))
# datetime.datetime(2012, 12, 16, 0, 0)

# 如果你需要执行大量的日期计算，最好安装第三方包python-dateutil
# import datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.rrule import *

d = datetime.now()
print(d)
# 2012-12-23 16:31:52.718111

# 返回当前月份开始日和下个月开始日组成的元组对象
from datetime import datetime,date,timedelta
import calendar

def get_month_range(start_date=None):
    if start_date is None:
        start_date = date.today().replace(day=1)
    _, days_in_month = calendar.monthrange(start_date.year, start_date.month)
    end_date = start_date + timedelta(days=days_in_month)
    return (start_date, end_date)

a_day = timedelta(days=1)
first_day, last_day = get_month_range()
while first_day < last_day:
    print(last_day)
    first_day += a_day

# 生成器函数难道是生成了一个可迭代的数组?
# 使用生成器函数来实现range
# python中的日期和时间能够使用标准的数字和比较操作符来进行运算    
def date_range(start, stop, step):
    while start < stop:
        yield start
        start += step
for d in date_range(datetime(2012, 9, 1), datetime(2012,10,1), timedelta(hours=6)):
    print(d)

# strftime可以很方便的对日期字符串做各种format操作，但是性能会差很多
from datetime import datetime
text = '2012-09-20'
y = datetime.strftime(text, '%Y-%m-%d')
z = datetime.now()
diff = z - y
# datetime.timedelta(3, 77824, 177393)
nice_z = datetime.strftime(z, '%A %B %d, %Y')
# 'Sunday September 23, 2012'
# 当处理大量的设计到日期的数据，并且format格式比较固定(如 YYYY-MM-DD)，可以考虑手动实现
from datetime import datetime
def parse_ymd(s):
    year_s, mon_s, day_s = s.split('-')
    return datetime(int(year_s), int(mon_s), int(day_s))
# 比使用datetime.strftime快七倍多

# python的时区和时区名称获取
from datetime import datetime

d = datetime(2012, 12, 21, 9, 30, 0)
# localize the date for Chicago
central = timezone('US/Central')
loc_d = central.localize(d)
print(loc_d)
# 2012-12-21 09:30:00-06:00
# 转化为其他时区的时间
bang_d = loc_d.astimezone(timezone('Asia/Kolkata'))
# 本地化日期上执行计算，需要考虑夏令时间等因素，为了防止此问题，可以使用normalize()方法
later = central.normalize(loc_d + timedelta(minutes=30))
print(later)
# 2013-03-10 03:15:00-05:00
# 为了防止被各种区域差别弄的晕头转向，处理本地化日期通常的策略现将所有日期转换为utc，并用它来执行所有的中间存储和操作
utc_d = loc_d.astimezone(pytz.utc)
print(utc_d)

# 获取时区名称
pytz.country_timezones['IN']
# ['Asia/Kolkata']


