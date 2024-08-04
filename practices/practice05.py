# 接收任意数量的参数的函数
def avg(first, *rest):
    return (first + sum(rest)) / (1 + len(rest))

# sample use
avg(1,2) 

# 创建标签属性
import html
# attrs是一个包含所有被传进来的关键字参数的字典
def make_element(name, value, **attrs):
    keyvals = [' %s="%s"' % item for item in attrs.items()]
    attr_str = ''.join(keyvals)
    # html.escape 是为了防xss注入吗？
    element = '<{name}{attrs}>{value}</{name}>'.format(
        name=name,
        attrs=attr_str,
        value=html.escape(value)
    )
    return element

# Example
# Creates '<item size="large" quantity="6">Albatross</item>'
make_element('item', 'Albatross', size='large', quantity=6)

# Creates '<p>&lt;spam&gt;</p>'
make_element('p', '<spam>')
# 接受任意数量的位置参数和关键字参数
def anyargs(*args, **kwargs):
    print(args) # a tuple
    print(kwargs) # a dict

# 所有位置参数会被放到args元组中，关键字参数会被放到字典kwargs中
# *参数只能出现在函数定义中最后一个位置参数后面，**参数只能出现在最后一个参数
def a(x, *args, y):
    pass
def b(x, *args, y, **kwargs):
    pass

# 只接受关键字参数的函数
def recv(maxsize, *, block):
    'Receives a message'
    pass
recv(1024, True) # TypeError
recv(1024, block=True) # ok

# 在接收任意多个位置参数的函数中指定关键字参数
def minimum(*values, clip=None):
    m = min(values)
    if clip is not None:
        m = clip if clip > m else m
    return m
minimum(1, 5, 2, -5, 10) # Returns -5
minimum(1, 5, 2, -5, 10, clip=0) # Returns 0
# 使用强制关键字参数可以使程序更具可读性，在使用函数help的时候输出也更容易理解
help(recv)
# Help on function recv in module __main__:
# recv(maxsize, *, block)
#     Receives a message

# 可以给函数添加类型注解方便理解和贴在文档中获生成接口文档？，尽管这些注解不会被类型检查
def add(x: int, y: int) -> int:
    return x + y
# 函数注解只存储在函数的 __annotations__属性中
add.__annotations__
# {'y': <class 'int'>, 'return': <class 'int'>, 'x': <class 'int'>}

# 实际上使用的是逗号来生成一个元组，而不是括号
# 返回多个值
def myfun():
    return 1, 2, 3
a, b, c =  myfun()
# a 1, b 2, c 3
d = myfun()
# (1, 2, 3)

# 带有默认值的可选参数
def spam(a, b=42):
    print(a, b)
spam(1) # Ok. a=1, b=42
spam(1,2) # Ok. a=1, b=2

# 默认参数是一个可修改的容器比如一个列表、集合或者字典，使用None作默认值
def spam(a, b=None):
    if b is None:
        b = []
# 如果你只是想测试某个默认参数是不是有传递进来,_no_value可以有效的判断调用时某个参数是否被传递进来了
_no_value = object()

def spam(a, b=_no_value):
    if b is _no_value:
        print('No b value supplied')
spam(1)
# No b value supplied
spam(1, 2) # b = 2
spam(1, None) # b = None

# 默认参数的值应该是不可变的对象，如果你传递数组，可能会遇到各种麻烦
def spam(a, b=[]):
    print(b)
    return b
x = spam(1)
x # []
x.append(99)
x.append('Yow!')
x # [99, 'Yow!']
spam(1) # [99, 'Yow!']


# 可以借宿lambda来定义匿名或内联函数
add = lambda x, y: x + y
add(2, 3) # 5
add('hello', 'world') # 'helloworld'
def add(x,y):
    return x + y
add(2,3) # 5
names = ['David Beazley', 'Brian Jones', 'Raymond Hettinger', 'Ned Batchelder']
sorted(names, key=lambda name: name.split()[-1].lower())
# ['Ned Batchelder', 'David Beazley', 'Raymond Hettinger', 'Brian Jones']
#lambda表达式典型的使用场景时排序或者数据reduce

x = 10
a = lambda y: x + y 
x = 20
b = lambda y: x + y 
a(10) # 30
b(10) # 30
# lambda表达式中的x是一个自由变量，在运行时绑定，而不是定义时绑定，这跟函数的默认值参数的定义是不同的
x = 15
a(10) # 25
x = 3
a(10) # 13
# 如果你想让某个匿名函数在定义时就捕获到值，可以将那个参数值定义成默认参数
x = 10
a = lambda y, x=x: x + y 
x = 20
b = lambda y, x=x: x + y 
a(10) # 20
b(10) # 30

# 期望函数在定义时就记住每次的迭代值
funcs = [lambda x: x+n for n in range(5)]
for f in funcs:
    print(f)

# 通过使用函数默认值参数形式，lambda函数在定义时就能绑定到值
funcs = [lambda x, n=n: x+n for n in range(5)]
for f in funcs:
    print(f)

# 如果一个函数有多个参数，可以使用function.partial() 函数来固定一些参数的值
def spam(a, b, c, d):
    print(a, b, c, d)

# 使用partial来固定某些参数值
from functools import partial
s1 = partial(spam, 1) # a = 1
s1(2,3,4)
# 1 2 3 4
s1(4,5,6)
# 1 4 5 6
s2 = partial(spam, d=42)
s2(1, 2, 3)
# 1 2 3 42
s3 = partial(spam, 1, 2, d =42)
s3(3)
# 1 2 3 42
s3(5)
# 1 2 5 42
# partial() 固定某些参数并返回一个新的callable对象，这个新的callable接受未赋值的参数
# 然后跟之前已经赋值过的参数合并起来，最后将所有参数传递给原始函数
# 计算两点间的距离
points = [(1,2),(3,4),(5,6),(7,8)]

import math
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)
# 通过固定参数的方式固定一个点，计算其他点与这个点的距离
pt = (4,3)
points.sort(key=partial(distance, pt))
points
# [(3,4),(1,2),(5,6),(7,8)]
# partial 通常用来微调其他库函数所使用的回调函数的参数
def output_result(result, log=None):
    if log is not None:
        log.debug('Got: %r', result)

# a sample function
def add(x, y):
    return x + y

if __name__ == '__main__':
    import logging
    from multiprocessing import Pool
    from functools import partial

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('test')
    p = Pool()
    p.apply_async(add, (3,4), callback=partial(output_result, log=log))
    p.close()
    p.join()


from socketserver import StreamRequestHandler, TCPServer

class EchoHandler(StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            self.wfile.write(b'GOT:' + line)

serv = TCPServer(('', 15000), EchoHandler)
serv.serve_forever()


# 你可以给echohandler增加一个可以接受其他配置选项的__init__方法
class EchoHandler(StreamRequestHandler):
    # ack is added keyword-only argument. *args, **kwargs are
    # any normal parameters supplied (which are passed on)
    def __init__(self, *args, ack, **kwargs):
        self.ack = ack
        super().__init__(*args, **kwargs)
    
    def handle(self):
        for line in self.rfile:
            self.wfile.write(self.ack + line)

# 使用partial来传递ack参数的初始值
from functools import partial
serv = TCPServer(('', 15000), partial(EchoHandler, ack=b'RECEIVED:'))
serv.serve_forever()

# 很多时候partial能实现的效果，lambda表达式也能实现，但是partial可以更加直观的表达你的意图
points.sort(key=lambda p: distance(pt, p))
p.apply_async(add, (3,4), callback=lambda result: output_result(result, log))
serv = TCPServer(('', 15000), 
                 lambda *args, **kwargs: EchoHandler(*args, ack=b'RECEIVED:', **kwargs))

# 将单方法的类转换为函数
from urllib.request import urlopen

class UrlTemplate:
    def __init__(self, template):
        self.template = template

    def open(self, **kwargs):
        return urlopen(self.template.format_map(kwargs))
# example use, download stock data from yahoo
yahoo = UrlTemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo.open(names='IPM,AAPL,FB', fields='sl1c1v'):
    print(line.decode('utf-8'))

# 这个类可以被一个更简单的函数代替
def urltemplate(template):
    def opener(**kwargs):
        return urlopen(template.format_map(kwargs))
    return opener

# example use
yahoo = urltemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo(names='IBM,AAPL,FB', fields='sl1c1v'):
    print(line.decode('utf-8'))
# 你可以通过使用内部函数或者闭包的方法来更优雅的保存需要额外保存的状态从而将单方法的类简化为一个函数
# 任何时候碰到需要给某个函数增加额外的状态信息的问题，都可以使用闭包，相比较讲一个函数转换为一个类而言，闭包通常是一种更加简洁和优雅的方案
def apply_async(func, args, *, callback):
    # compute the result
    result = func(*args)
    # invoke the callback with  the result
    callback(result)

def print_result(result):
    print('Got:', result)

def add(x, y):
    return x + y

apply_async(add, (2, 3), callback=print_result)
# Got: 5
apply_async(add, ('hello', 'world'), callback=print_result)
# Got: helloworld

# 想让回调函数访问其他变量或者特定环境的变量值
class ResultHandler:
    def __init__(self):
        self.sequence = 0
    def handler(self, result):
        self.sequence += 1
        print('[{}] Got: {}'.format(self.sequence, result))

r = ResultHandler()
apply_async(add, (2,3), callback=r.handler)
# [1] Got: 5
apply_async(add, ('hello', 'world'), callback=r.handler)
# [2] Got: helloworld

    # 使用一个闭包捕获状态值
def make_handler():
    sequence = 0
    def handler(result):
        # nonlocal用来指示接下来的变量会在回调函数中被修改，没有这个声明，代码会报错
        nonlocal sequence
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
    return handler

handler = make_handler()
apply_async(add, (2,3), callback=handler)
# [1] Got: 5
apply_async(add, ('hello', 'world'), callback=handler)
# [2] Got: helloworld

def make_handler():
    sequence = 0
    while True:
        result = yield
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
handler = make_handler()
next(handler) # advance to the yield
apply_async(add, (2,3), callback=handler.send)
# [1] Got: 5
apply_async(add, ('hello', 'world'), callback=handler.send)
# [2] Got: helloworld

# 还可以借助lambda 和 partial的方式来给回调函数传递额外的值
apply_async(add, (2, 3), callback=lambda r: handler(r, seq))
# [1] Got: 5

# 通过使用生成器和协程使得回调函数内联在某个函数中
def apply_async(func, args, *, callback):
    # compute the result
    result = func(*args)
    # invoke the callback with the result
    callback(result)

# Async类和inlined_async
from queue import Queue
from functools import wraps

class Async:
    def __init__(self, func, args):
        self.func = func
        self.args = args
    
    def inlined_async(func):
        @wraps(func)
        def wrapper(*args):
            f = func(*args)
            result_queue = Queue()
            result_queue.put(None)
            while True:
                result = result_queue.get()
                try:
                    a = f.send(result)
                    apply_async(a.func, a.args, callback=result_queue.put)
                except StopIteration:
                    break
        return wrapper
# 上述两个代码片段允许我使用yield语句联调回调步骤

def add(x,y):
    return x + y
@inlined_async
def test():
    r = yield Async(add, (2, 3))
    print(r)
    for n in range(10):
        r = yield Async(add, (n,n))
        print(r)
    print('Goodbye')

test()

if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool()
    apply_async = pool.apply_async
    # run the test function
    test()

# 将复杂的控制流隐藏到生成器函数背后的例子在标准库和第三方保重都能看到
# 在contextlib中的@contextmanager装饰器使用了一个令人费解的技巧，通过一个yield语句将进入和离开上下文管理器粘合在一起
# nonlocal声明可以让我们编写函数来修改内部变量的值
def sample():
    n = 0
    # closure function
    def func():
        print('n=', n)
    
    # accessor methods for n
    def get_n():
        return n
    def set_n(value):
        nonlocal n
        n = value

    func.get_n = get_n
    func.set_n = set_n
    return func
# 函数属性允许我们用一种很简单的方式将访问方法绑定到闭包函数上，这个跟实例方法很像
import sys
class ClosureInstance:
    def __init__(self, locals=None):
        if locals is None:
            locals = sys._getframe(1).f_locals
        # update instance dictionary with callables
        # 更新items的值到字典里
        self.__dict__.update((key, value) for key, value in locals.items() if callable(value))

    def __len__(self):
        return self.__dict__['__len__']()

# example use
def Stack():
    items = []
    def push(item):
        items.append(item)
    def pop():
        return items.pop()
    def __len__():
        return len(items)
    return ClosureInstance()

s = Stack()
s.push(10)
s.push(20)
s.push('hello')
len(s) # 3
s.pop()
# 'Hello'
s.pop()
# 20
s.pop()
# 10
# 定义一个普通的类
class Stack2():
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def __len__(self):
        return len(self.items)
    
# 函数的变量状态维护可以通过闭包的方式实现，也可以使用类然后定义类的属性
# 自定义__repr__和__str通常是很好的习惯，因为它能简化调试和实例输出，如果仅仅只是打印输出或日志输出某个实例
# 如果__str__()没有被定义，就会使用__repr__来代替输出
# 改变实例的字符串表示，可重新定义它的__str__() 和 __repr__方法
class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # !r格式化代码指明输出使用__repr__来代替__str__
    # {0.x}对应的是第1个参数的x属性，0实际上指的self本身
    def __repr__(self):
        return 'Pair({0.x!r}, {0.y!r})'.format(self)
        # 作为替代，可以使用%操作符号
        # return 'Pair(%r, %r)' % format(self.x, self.y)

    def __str__(self):
        return '({0.x!s}, {0.y!s})'.format(self)
p = Pair(3, 4)
# Pair(3, 4) # __repr__() output
print(p)
# (3,4) # __str__() output