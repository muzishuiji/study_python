# 支持自定义字符串的格式化，想要在类上面定义__format__方法
_formats = {
    'ymd': '{d.year}-{d.month}-{d.day}',
    'mdy': '{d.month}/{d.day}/{d.year}',
    'dmy': '{d.day}/{d.month}/{d.year}',
}
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    def __format__(self, code):
        if code == '':
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)
d = Date(2012, 12, 21)
format(d)  # '2012-12-21'
format(d, 'mdy') # '12/21/2012'
'The date is {:ymd}'.format(d) # 'The date is 2012-12-21'
'The date is {:mdy}'.format(d) # 'The date is 12/21/2012'

# 格式化代码可以是任何值
from datetime import date
from typing import Any
d = date(2012, 12, 21)
format(d) # '2012-12-21'
format(d, '%A, %B %d, %Y')
# 'Friday, December 21, 2012'
# 对于内置类型的格式化有一些标准的约定，可参考string模块文档说明
'The date is {:%d %b %Y}. Goodbye'.format(d) # 'The date is 21 Dec 2012. Goodbye'

# 让一个对象兼容with语句，需要实现__enter__ 和__exit__方法
from socket import socket, AF_INET, SOCK_STREAM

class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = family
        self.type = type
        self.sock = None
    def __enter__(self):
        if self.sock is not None:
            raise RuntimeError('Already connected')   
        self.sock = socket(self.family, self.type) 
        self.sock.connect(self.address)
        return self.sock
    # exc_ty: 异常类型，exc_val：异常值，tb：追溯
    # __exit__ 决定如何利用一场戏逆袭，返回true则异常会被清空，with语句后面的程序正常执行
    def __exit__(self, exc_ty, exc_val, tb):
        self.sock.close()
        self.sock  = None

# sock连接的建立和关闭都是用with语句自动完成的
from functools import partial
conn = LazyConnection('www.python.org', 80)
# connection closed
with conn as s:
    # conn.__enter__() executes: connection open
    s.send(b'GET /index.html HTTP/1.0\r\n')
    s.send(b'Host: www.python.org\r\n')
    s.send(b'\r\n')
    resp = b''.join(iter(partial(s.serv, 8192), b''))
    # conn.__exit__() executes: connection closed

# 多个with语句的嵌套使用可以像下面这样
from socket import socket, AF_INET, SOCK_STREAM

class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = family
        self.type = type
        self.connections = []
    def __enter__(self):
        sock = socket(self.family, self.type)
        sock.connect(self.address)
        self.connections.append(sock)
        return sock
    def __exit__(self, exc_ty, exc_val, tb):
        self.connections.pop().close()
    
# Example use
from functools import partial
conn = LazyConnection(('www.python.org'), 80)
with conn as s1:
    pass
    with conn as s2:
        pass
        # s1,s2 are independent sockets

# 在需要管理一些资源如文件、网络连接和锁的编程环境中，使用上下文管理器是很普遍的
# 这些资源的一个主要特征是它们必须被手动的关闭或释放来确保程序的正确运行
# 如果你请求了一个锁，那么你必须确保之后释放了它
# 当你的程序需要创建大量的对象，占用很大的内存时，推荐使用slots属性
class Date:
    __slots__ = ['year', 'month', 'day']
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
# 定义slots后，python就会为实例使用一种更加紧凑的内部表示，实例通过一个很小的固定大小的数组来构建，而不是为每个实例定义一个字典，这跟元组或列表很类似
# 使用slots一个不好的地方就是不能再给实例添加新的属性了
# 使用slot会导致创建实例的灵活性下降，你应该只用在经常被使用到的用作数据结构的类上定义slots，比如在程序中需要创建某个类的几百万个实例对象
# slots也可以作为一个封装工具来防止用户给实例增加新的属性
# 如果你定义的一个变量和某个保留关键字冲突，可以使用单下划线作为后缀
lambda_ = 2.0
# 单下划线定义私有属性
class A:
    def __init__(self):
        self._internal = 0
        self.public = 1
    def public_method(self):
        '''
        A public method
        '''
        pass
    def _internal_method(self):
        pass

# 双下划线命名的私有属性，子类无法通过继承来实现覆盖
class B:
    def __init__(self):
        self.__private = 0
    def __private_method(self):
        pass
    def public_method(self):
        pass
        self.__private_method()
class C(B):
    def __init__(self):
        super().__init__()
        self.__private = 1 # 不会覆盖B.__private
    # Does not override b.__private_method()
    def __private_method(self):
        pass
# 私有属性__private, __private_method被重命名为_C_private 和 _C_private_method
# 双下划线命名的私有属性，子类无法通过继承来实现覆盖
c_instance= C()

attributes = dir(c_instance)
print(attributes)
# 若你不希望你的私有属性被子类覆盖，则使用双划线，保证你的私有属性在子类中隐藏

# 通过property来自定义某个属性
class Person:
    def __init__(self, first_name):
        self._first_name = first_name
    
    # getter function
    @property
    def first_name(self):
        return self._first_name
    # setter function
    @first_name.setter
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Excepted a string') 
        self._first_name = value
    # deter function(optional)
    @first_name.deleter
    def first_name(self):
        raise AttributeError('can not delete attribute')
# property的一个关键特征是它看上去和普通的attribute没什么两样，但是访问它的时候会自动触发getter，setter，deleter
a = Person('Guido')
a.first_name # Guido
a.first_name = 42
# throw error
del a.first_name
# throw error

# 在已存在的get和set方法基础上定义property
class Person:
    def __init__(self, first_name):
        self.set_first_name(first_name)
    # getter function
    def get_first_name(self):
        return self._first_name
    # setter function
    def set_first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Excepted a string') 
        self._first_name = value
    # deleter function(optional)
    def del_first_name(self):
        raise AttributeError('can not delete attribute')
    # make a property from existing get/set methods
    name = property(get_first_name, set_first_name, del_first_name)
    # 一个property属性其实就是一系列相关绑定方法的集合，这些方法不会被直接触发，而是会在访问property时自动触发
# properties 还支持动态计算attribute的方法
import math
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @property
    def area(self):
        return math.pi * self.radius ** 2
    @property
    def diameter(self):
        return self.radius * 2
    @property
    def perimeter(self):
        return 2 * math.pi * self.radius
# 针对多个属性分别定义get和set方法：重复代码会导致臃肿，移除错和丑陋的程序
# 在python中，通过使用装饰器或闭包，可以有更好的方法完成同样的事情
class A:
    def spam(self):
        print('A.spam')
class B:
    def __init__(self):
        super().__init__()
        self.y = 1

# super也可以用于特殊情况调用父类的方法，其他情况调用自身的逻辑
# 即使没有显式的指明某个类的父类，Proxy(Base) ，super仍然可以有效的工作
class Proxy:
    def __init__(self, obj):
        self._obj = obj
    # delegate attribute lookup to internal obj
    def __getattr__(self, name):
        return getattr(self._obj, name)
    # delegate attribute assignment
    def __setattr__(self, name: str, value: Any):
        # 私有属性绑定在父类上？
        if(name.startswith('_')):
            super().__setattr__(name, value)
        else:
            setattr(self._obj, name, value)
# 你也可以直接调用父类的方法，但是这可能会导致某个父类方法的多次调用
class Base:
    def __init__(self):
        print('Base.__init__')
class A(Base):
    def __init__(self):
        Base.__init__(self)
        print('A.__init__')
class B(Base):
    def __init__(self):
        Base.__init__(self)
        print('B.__init__')
class C(A, B):
    def __init__(self):
        A.__init__()
        B.__init__()
        print('C.__init__')
# Base.__init__()会被调用两次，换成super则不会
class Base:
    def __init__(self):
        print('Base.__init__')
class A(Base):
    def __init__(self):
        super().__init__(self)
        print('A.__init__')
class B(Base):
    def __init__(self):
        super().__init__(self)
        print('B.__init__')
class C(A, B):
    def __init__(self):
        super().__init__()
        print('C.__init__')
# Python中会对定义的类计算出一个所谓的方法解析顺序（MRO）列表，这个MRO列表就是一个简单的所有基类的线性顺序表
# C.__mro__
# (<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>,
# <class '__main__.Base'>, <class 'object'>)
# MRO列表生成遵循如下三条准则
# - 子类会先于父类被检查
# - 多个父类根据他们在列表中的顺序被检查
# - 如果对下一个类存在两个合法的选择，选择第一个父类
# 使用super的时候需要确保所有继承的父类中确实有对应方法，不然会报错

# 父类Person，定义了一个property
class Person:
    def __init__(self, name):
        self.name = name
    # getter function
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._name = value
    
    # deleter function
    @name.deleter
    def name(self):
       raise AttributeError("Can't delete attribute")

# 继承Person并扩展name属性的功能
class SubPerson(Person):
    @property
    def name(self):
        print('Getting name')
        return super().name
    @name.setter
    def name(self, value):
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)
    @name.deleter
    def name(self):
        print('Deleting name')
        # 使用类变量而不是实例变量来访问它
        super(SubPerson, SubPerson).name.__delete__(self)

# 若只想扩展property的某一个方法，则这样写
class SubPerson(Person):
    @Person.name.getter
    def name(self): 
        print('Getting name')
        return super().name

class SubPerson(Person):
    @Person.name.setter
    def name(self, value): 
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value) 

# 扩展一个描述器: 描述器就用来定义类的property的各种描述方法？？？
# 我懂了，就是可以很方便的定义各种不同的属性，而不是只是'name'，可以是"work", "class"等属性
# 描述器可以用于批量生成不同的property
class String:
    def __init__(self, name):
        self.name = name
    
    def __get__(self, instance, cls):
        if instance is None:
            return self
        return instance.__dict__[self.name]
    
    def __set__(self, instance, value):
        if not isinstance(value, str):
           raise TypeError('Expected a string')
        instance.__dict__[self.name] = value
# a class with a descriptor
class Person:
    name = String('name')

    def __init__(self, name):
        self.name = name
    
# extending a descriptor with a property
class SubPerson(Person):
    @property
    def name(self):
        print('Getting name')
        return super().name
    
    @name.setter
    def name(self, value):
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        print('Deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)    
        
# 如果你只是想简单的自定义某个类的单个属性访问的话就不用去写描述器了
# 当程序中有很多重复代码的时候描述器就很有用了
# 创建一个数字类型的属性的描述器
class Integer:
    def  __init__(self, name):
        self.name = name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]
    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError('Expected an int')
        instance.__dict__[self.name] = value
    def __delete__(self, instance):
        del instance.__dict__[self.name]
        
# 描述器的使用
class Point:
    x = Integer('x')
    y = Integer('y')

    def __init__(self, x, y):
        self.x = x
        self.y = y
# 描述器可实现大部分python类特性中的底层魔法，包括@classmethod，@staticmethod，@property，甚至__slots__特性
# 描述器一个比较困惑的地方是它只能在类级别被定义，而不能为每个实例单独定义
class Point:
    def __init__(self, x, y):
        self.x = Integer('x') # no, must be a class variable
        self.y = Integer('y')
        self.x = x
        self.y = y
# 描述器通常是那些使用装饰器或元类的大型框架中的一个组件
# 更高级的装饰器
# descriptor for a type-checked attribute
class Typed:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('Expected ' + str(self.expected_type))
        instance.__dict__[self.name] = value
    def __delete__(self, instance):
        del instance.__dict__[self.name]
# class decorator that applies it to selected attributes
# 类装饰器：用它来筛选属性
def typeassert(**kwargs):
    def decorate(cls):
        for name, expected_type in kwargs.items():
            #设置一个类型装饰器给这个类
            setattr(cls, name, Typed(name, expected_type))
        return cls
    return decorate

# 使用示例
@typeassert(name=str, shares=int, prince=float)
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price
# 如果你只是简单的自定义某个类的单个属性访问的话就不用写描述器了，使用property技术会更容易
# 当程序中有很多重复代码的时候描述器就很有用了，比如你想在代码的很多地方使用描述器提供的功能或者将它作为一个函数库特性
# 定义一个延迟属性的一种高效方法是通过使用一个描述器类
# 如果一个描述器只定义了get方法的话，它通常具有更弱的绑定，当被访问属性不在实例底层的字典中时，底层的字典中__get__方法才会被触发
# lazyproperty利用这一点，使用__get__()方法在实例中存储计算出来的值，这个实例使用相同的名字作为它的property，这样一来，结果值被存储在字典实例中并且以后就不需要去计算这些property了

class lazyproperty:
    def __init__(self, func):
        self.func = func
    
    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value
# 在类中使用
import math

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @lazyproperty
    def area(self):
        print('Computing area')
        return math.pi * self.radius ** 2
    @lazyproperty
    def perimeter(self):
        print('Computing perimeter')
        return 2 * math.pi * self.radius
    
c = Circle(4.0)
c.radius # 4.0
c.area 
# Computing area
# 50.26548245743669
c.perimeter
# Computing perimeter
# 25.132741228718345
c.perimeter
# 25.132741228718345
# 有一个小缺陷是计算出的值被创建后是可以被修改的
c.area
# Computing area
# 50.26548245743669
c.area = 25
c.area # 25
# 像下面这样修改，则不可修改
def lazyproperty(func):
    name = '_lazy_' + func.__name__
    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value
    return lazy
c.area = 25
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
# AttributeError: can't set attribute
# 这种方案的一个缺点就是所有get操作都必须被重定向到属性的getter函数上去，这个相比在实例字典中查找值的方案效率要低一点

# 打印两个help(Stock)
# 基类中定义一个公用的__init__函数
import math
class Structure1:
    # class variable that specifies excepted fields
    _fields = []

    def __init__(self, *args):
        if(len(args) != len(self._fields)):
            raise TypeError('Excepted {} arguments'.format(len(self._fields)))
        # Set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
# 继承
class Stock(Structure1):
    _fields = ['name', 'shares', 'price']

class Point(Structure1):
    _fields = ['x', 'y']

class Circle(Structure1):
    _fields = ['radius']
    def area(self):
        return math.pi * self.radius ** 2
s = Stock('ACME', 50, 91.1)
p = Point(2, 3)
c = Circle(4.5)
s2 = Stock('ACME', 50)
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "structure.py", line 6, in __init__
#         raise TypeError('Expected {} arguments'.format(len(self._fields)))
# TypeError: Expected 3 arguments

# 如果你想支持关键字参数，可以将关键字参数设置为实例属性
class Structure2:
    _fields=[]

    def __init__(self, *args, **kwargs):
        if(len(args) > len(self._fields)):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        # set all of the positional arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
        print(len(args), '=========+')
        # 设置剩下的关键字参数
        for name in self._fields[len(args):]:
            print(name, '=========')
            print(self._fields, '====')
            print(self._fields[len(args):], '+++')
            setattr(self, name, kwargs.pop(name))
        # 检查是否有剩余的未知参数
        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))
# example
if __name__ == '__main__':
    class Stock(Structure2):
        _fields = ['name', 'shares', 'price']
    s1 = Stock('ACME', 50, 91.1)
    s2 = Stock('ACME', 50, shares=91.1)
    s3 = Stock('ACME', shares=50, price=91.1)

# *args：加一个星号*在参数前，表示该参数是一个可变位置参数，可以接受任意多的非关键字参数，这些参数会以一个元组的形式传入
def foo(*args):
    for arg in args:
        print(arg)
foo(1,2,3,4) 
# **kwargs：表示该参数是一个可变关键字参数，可以接受任意多的关键字参数，这些参数会以一个字典的形式传入
def bar(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")
bar(a=1,b=2,c=3) # 输出：a = 1, b = 2, c = 3

# 将不在_fields中的名称加入到属性中去
class Structure3:
    _fields=[]

    def __init__(self, *args, **kwargs):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        # set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
        # 设置自定义的参数
        extra_args = kwargs.keys() - self._fields
        for name in extra_args:
            setattr(self, name, kwargs.pop(name))
        
        if kwargs:
            raise TypeError('Duplicate values for {}'.format(','.join(kwargs)))
# example use
if __name__ == '__main__':
    class Stock(Structure3):
        _fields = ['name', 'shares', 'price']

    s1 = Stock('ACME', 50, 91.1)
    s2 = Stock('ACME', 50, 91.1, date='8/2/2012')

# 不想使用setattr函数设置属性值，而想直接更新实例字典
class Structure:
    _fields = []
    def __init__(self, *args):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))
        # 直接更新实例字典
        self.__dict__.update(zip(self._fields, args))
if __name__ == '__main__':
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']
# 当一个子类定义了__slots__或者通过property来包装某个属性，直接访问字典就不起作用了
# setattr是更通用的写法

# 定义接口或抽象类，并通过执行类型检查来确保zi lei子类实现了某些特定的方法
from abc import ABCMeta, abstractmethod

class IStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxbytes=-1):
        pass

    @abstractmethod
    def write(self, data):
        pass
# 抽象类的特点是不能直接被实例化
# 抽象类的目的就是让别的类继承它并实现特定的抽象方法
class SocketStream(IStream):
    def read(self, maxbytes=-1):
        pass
    def write(self, data):
        pass
# 抽象基类的主要用途是在代码中见擦好某些类是否为特定类型，实现了特定接口
def serialize(obj, stream):
    # - 检查是否为特定类型
    # - 检查是否实现了特定接口
    if not isinstance(stream, IStream):
        raise TypeError('Expected an IStream')
    pass

#  还可以通过注册方式来让某个类实现抽象基类
import io
IStream.register(io.IOBase)
# open a normal file and type check
f = open('foo.txt')
isinstance(f, IStream)

# abstractmethod还能注解静态方法，类方法和properties
class A(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        pass

    @name.setter
    @abstractmethod
    def name(self, value):
        pass

    @classmethod
    @abstractmethod
    def method1(cls):
        pass

    @staticmethod
    @abstractmethod
    def method2():
        pass
# 标准库中也用到了很多抽象类
# collections模块定义了很多跟容器和迭代器有关的抽象基类，numbers库定义了跟数字对象有关的基类
# io库定义了很多跟io操作相关的基类
import collections
# check is x is a sequence
if isinstance(x, collections.Sequence):
    #...
# check is x is iterable
if isinstance(x, collections.Iterable):
    #...
# check if x has a size
if isinstance(x, collections.Sized):
    #...
# check is x is a mapping
if isinstance(x, collections.Mapping):
    #...

# 使用描述器来实现自定义属性赋值函数
# 系统类型和赋值验证框架
# base class，uses a descriptor to set a value
class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        for key, value in opts.items():
            setattr(self, key, value)
    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

# descriptor for enforcing types
class Typed(Descriptor):
    excepted_type = type(None)

    def __set__(self, instance, value):
        if not isinstance(value, self.excepted_type):
            raise TypeError('expected ' + str(self.expected_type)) 
        return super().__set__(instance, value)

# descriptor for enforcing values
# 约束set value >= 0
class Unsigned(Descriptor):
    def __set__(self, instance, value):
        if(value < 0):
            raise ValueError('Expected >= 0')
        return super().__set__(instance, value)

# 约束value size
class MaxSized(Descriptor):
    # opts 用来接受关键字参数
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        return super().__init__(name, **opts)

    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        return super().__set__(instance, value)

# 实际定义的各种不同的数据类型
# 所有描述器类都是混入类来实现的，比如Unsigned和MaxSized都要跟其他继承自Typed类混入
# 这里利用多继承来实现相应的功能
class Integer(Typed):
    excepted_type = int

class UnsignedInteger(Integer, Unsigned):
    pass

class Float(Typed):
    excepted_type = float

class UnsignedFloat(Float, Unsigned):
    pass

class String(Typed):
    excepted_type = str

class SizedString(String, MaxSized):
    pass

# 使用这些自定义数据类型定义类
class Stock:
    # specify constraints
    name = SizedString('name', size=8)
    shares = UnsignedInteger('shares')
    price = UnsignedFloat('price')

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

s.name # 'ACME'       
s.shares = 75
s.shares = -10
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "example.py", line 17, in __set__
#         super().__set__(instance, value)
#     File "example.py", line 23, in __set__
#         raise ValueError('Expected >= 0')
# ValueError: Expected >= 0
s.price = 'a lot'
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "example.py", line 16, in __set__
#         raise TypeError('expected ' + str(self.expected_type))
# TypeError: expected <class 'float'>
s.name = 'ABRACADABRA'
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "example.py", line 17, in __set__
#         super().__set__(instance, value)
#     File "example.py", line 35, in __set__
#         raise ValueError('size must be < ' + str(self.size))
# ValueError: size must be < 8

# 还有一些技术可以简化上面的代码，使用类装饰器
def check_attributes(**kwargs):
    def decorate(cls):
        for key, value in kwargs.items():
            if isinstance(value, Descriptor):
                value.name = key
                setattr(cls, key, value)
            else:
                setattr(cls, key, value(key))
        return cls
    return decorate
# example
@check_attributes(name=SizedString(size=8),
                  shares=UnsignedInteger,
                  price=UnsignedFloat)
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

# a metaclass that applies checking
class checkedmeta(type):
    def __new__(cls, clsname, bases, methods):
        # attach attributes names to descriptors
        for key, value in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(cls, clsname, bases, methods)
# example
class Stock2(metaclass=checkedmeta):
    name = SizedString(size=8)
    shares = UnsignedInteger()
    price = UnsignedFloat()

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

# 混入类一个比较难理解的地方是，调用super函数时，你并不知道究竟调用哪个具体类
# 你需要跟其他类结合后才能正确的使用
# 使用类装饰器和元类通常可以简化代码
# Normal
class Point:
    x = Integer('x')
    y = Integer('y')

# Metaclass
class Point(metaclass=checkedmeta):
    x = Integer()
    y = Integer()
# 所有方法中，类装饰器方案应该是最灵活和最高明的，装饰器可以很容易的添加和删除
# 装饰器还能作为混入类的替代技术来实现同样的效果
# decorator for applying type checking
def Typed(excepted_type, cls=None):
    if cls is None:
        return lambda cls: Typed(excepted_type, cls)
    super_set = cls.__set__

    def __set__(self, instance, value):
        if not isinstance(value, excepted_type):
           raise TypeError('expected ' + str(excepted_type))
        super_set(self, instance, value) 
    cls.__set__ = __set__
    return cls

# Decorator for unsigned values
def Unsigned(cls):
    super_set = cls.__set__
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Expected >= 0')
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

# Decorator for allowing sized values
def MaxSized(cls):
    super.__init__ = cls.__init__
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        super_init(self, name, **opts)
    cls.__init__ = __init__
    super_set = cls.__set__
    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls
# specialized descriptors
@Typed(int)
class Integer(Descriptor):
    pass

@Unsigned
class UnsignedInteger(Integer):
    pass

@Typed(float)
class Float(Descriptor):
    pass

@Unsigned
class UnsignedFloat(Float):
    pass

@Typed(str)
class String(Descriptor):
    pass

@MaxSized
class SizedString(String):
    pass

# 你想实现一个自定义的类来模拟内置的容器类功能，比如列表和字典
# collections定义了很多抽象基类，当你需要对应的功能的时候，继承对应的类即可
import collections
class A(collections.Iterable):
    # 需要实现collections.Iterable的所有的抽象方法，否则会报错
    def __iter__(self) -> io.Iterator:
        return super().__iter__()
    pass

# 你可以试着实例化一个对象，在错误提示中找到可以实现哪些方法
import collections
collections.Sequence()

# 继承sequence抽象类，实现元素按顺序存储
class SortedItems(collections.Sequence):
    def __init__(self, initial=None):
        self.items = sorted(initial) if initial is not None else []
    # required sequence methods
    def __getitem__(self, index):
        return self.items[index]
    # required sequence methods
    def __len__(self) -> int:
        return len(self.items)
    # method for adding an item in the right location
    def add(self, item):
        bisect.insort(self.items, item)

items = SortedItems([5,1,3])
print(list(items))
print(items[0], items[-1])
items.add(2)
print(list(items))
# bisect: 是一个在排序列表中插入元素的高效方式，可以保证在元素插入后还保持顺序

# collections 中很多抽象类会为一些常见容器操作提供默认的实现，这样你只需要实现你感兴趣的方法即可
# 假设你的类继承自collections.MutableSequence
import collections
class Items(collections.MutableSequence):
    def __init__(self, initial=None):
        self._items = list(initial) if initial is not None else []
    # required sequence methods
    def __getitem__(self, index):
        print('Getting:', index)
        return self._items[index]

    def __setitem__(self, index, value):
        print('Setting:', index, value)
        self._items[index] = value

    def __delitem__(self, index):
        print('Deleting:', index)
        del self._items[index]
    def insert(self, index, value):
        print('Inserting:', index)
        self._items.insert(index, value)
    # required sequence methods
    def __len__(self):
        print('Len')
        return len(self._items)
# 创建出来的items实例，支持几乎所有的核心方法列表，如append(), remove(), count()等
# numbers模块提供了一个类似的跟整数类型相关的抽象类型集合，可尝试构建更多自定义抽象基类
a = Items([1,2,3])
a.append(4)
a.append(2)
a.remove(3)

    
# 简单的代理
class A:
    def spam(self, x):
        pass
    def foo(self):
        pass

class B1:
    '''简单的代理'''
    def __init__(self):
        self._a = A()

    def spam(self, x):
        # Delegate to the internal self._a instance
        return self._a.spam(x)
    
    def foo(self):
        # Delegate to the internal self._a instance
        return self._a.foo()
    def bar(self):
        pass
# 如果你有大量的方法需要代理，使用getattr方法会更好，该方法会在访问的attribute不存在时被调用
class B2:
    '''使用__getattr__的代理，代理方法比较多时候'''
    def __init__(self):
        self._a = A()
    def bar(self):
        pass
    # expose all of the methods defined on class A
    def __getattr__(self, name):
        return getattr(self._a, name)

b = B()
b.bar()
b.spam(42)

# Proxy类可以理解为对原先类的属性访问的隔离和方法的功能扩展（如加入日志功能，只读访问等）
# a proxy class that wraps around another object, but exposes its public attributes
class Proxy:
    def __init__(self, obj):
        self._obj = obj

    # delegate attribute lookup to internal obj
    def __getattr__(self, name):
        print('getattr:', name)
        return getattr(self._obj, name)
    
    # delegate attribute assignment
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            print('setattr:', name, value)
            setattr(self._obj, name, value)
    # delegate attribute deletion
    def __delattr__(self, name: str):
        if name.startswith('_'):
            super().__delattr__(name)
        else:
            print('delattr:', name)
            delattr(self._obj, name)
# 使用方式：用Proxy来包装其他类生成的实例
class Spam:
    def __init__(self, x):
        self.x = x
    
    def bar(self, y):
        print('Spam.bar', self.x, y)
# create an instance
s = Spam(2)
# create a proxy around it
p = Proxy(s)
# access the proxy
print(p.x)  # Outputs 2
p.bar(3)  # Outputs "Spam.bar: 2 3"
p.x = 37  # Changes s.x to 37

# 代理类有时候可以作为继承的替代方案
class A:
    def spam(self, x):
        print('A.spam', x)
    def foo(self):
        print('A.foo')
class B:
    def spam(self, x):
        print('B.spam', x)
        super().spam(x)
    def bar(self):
        print('B.bar')

# 使用代理的话
class A:
    def spam(self, x):
        print('A.spam', x)
    def foo(self):
        print('A.foo')
class B:
    def __init__(self):
        self._a = A()
    def spam(self, x):
        print('B.spam', x)
        self._a.spam(x)
    def bar(self):
        print('B.bar')  
    def __getattr__(self, name):
        return getattr(self._a, name)      
# __getattr__会优先从代理类本身查询，查不到则会调用被代理的类的属性
# __setattr__和__delattr__需要额外的魔法来区分代理实例和被代理实例的_obj的属性
# __getattr__对于大部分双下划线开头和结尾的属性并不适用，需要自己去重定义
class ListLike:
    def __init__(self):
        self._items = []    

    def __getattr__(self, name):
        return getattr(self._items, name)
a = ListLike()
a.append(2)
a.insert(0, 1)
a.sort()
len(a)
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
# TypeError: object of type 'ListLike' has no len()
a[0]
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
# TypeError: 'ListLike' object does not support indexing

# 为了让ListLike创建的实例支持相关方法，你必须手动实现这些方法代理
class ListLike:
    def __init__(self):
        self._items = []
    
    def __getattr__(self, name):
        return getattr(self._a, name)      

    # added special methods to support certain list operations
    def __len__(self):
        return len(self._items)
    # 自定义查询方法
    def __getitem__(self, index):
        return self._items[index]
    # 自定义set方法
    def __setitem__(self, index, value):
        self._items[index] = value
    # 自定义delete方法
    def __delitem__(self, index):
        del self._items[index]

# 在类中定义多个构造器
import time
class Date:
    '''方法一：使用类方法'''
    # primary constructor
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    # alternate constructor
    # cls 指向当前类，最终的表现是创建一个当前类的实例化对象
    @classmethod
    def today(cls):
        t = time.localtime()
        return cls(t.tm_year, t.tm_mon, t.tm_mday)
    
# primary 初始化
a = Date(2012, 12, 21)
# alternate 初始化
b = Date.today()
# 定义多个构造器接受一个class作为第一个参数cls，这个类被用来创建并返回最终的实例
# 所以这样定义的类会继承传入的类
# 继承时
class NewDate(Date):
    pass
c = Date.today() # creates an instance of Date(cls=Date)
d = NewDate.today() # creates an instance of NewDate(cls=NewDate)

# 创建一个绕过__init__方法的实例
# 可以通过__new__方法来创建一个未初始化的实例
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
d = Date.__new__(Date)
# 这时需要手动初始化Date实例的属性
d.year
# >>> d.year
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
# AttributeError: 'Date' object has no attribute 'year'
data = {'year': 2012, 'month': 8, 'day': 29 }
for key, value in data.items():
    setattr(d, key, value)
d.year # 2012

# 定义一个新的沟槽函数today
from time import localtime

class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    @classmethod
    def today(cls):
        d = cls.__new__(cls)
        t = localtime()
        d.year = t.tm_year
        d.month = t.tm_mon
        d.day = t.tm_mday
        return d
# 这样，你在反序列化json数据时产生如下的字典对象
data = { 'year': 2012, 'month': 8, 'day': 29 }
# 如果你想把它转换成一个Date类型实例，可以用上面的技术
# 当你通过非常规方式创建实例的时候，最好不要直接去访问底层字典实例，除非你真的清楚所有细节
# 否则，如果这个类使用了__slots__，properties，descriptor或其他高级技术的时候代码会失效，
# 这时使用setattr方法会让你的代码更加通用
# 假设你想扩展映射对象，给它们添加日志、唯一性设置，类型检查等功能，可以使用混入类

# 继承LoggedMapping并混入一些日志打印
class LoggedMappingMixin:
    '''
    Add logging to get/set/delete operations for debugging
    '''
    __slots__ = () # 混入类么有实例变量，因为直接实例化混入类没有任何意义
    def __getitem__(self, key):
        print('Getting ' + str(key))
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        print('Setting {} = {!r}'.format(key, value))
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        print('Deleting ' + str(key))
        return super().__delitem__()

class SetOnceMappingMixin:
    '''
    Only allow a key to be set once
    '''
    __slots__ = ()

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(str(key) + ' already set')
        return super().__setitem__(key, value)

class StringKeyMappingMixin:
    '''
    Restrict keys to strings only
    '''
    __slots__ = ()

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('keys must be strings')
        return super().__setitem__(key, value)

# 这些类可以做多即成和其他映射对象混入使用
class LoggedDict(LoggedMappingMixin, dict):
    pass

d = LoggedDict()
d['x'] = 23
print(d['x'])
del d['x']

from collections import defaultdict
# 它们是用来通过多继承和其他映射对象混入使用的
class SetOnceDefaultDict(SetOnceMappingMixin, defaultdict):
    pass

d = SetOnceDefaultDict(list)
d['x'].append(2)
d['x'].append(3)
# d['x'] = 23  # KeyError: 'x already set'
# 混合类通常和已有的类结合，才能发挥正常功效，
# socketserver模块中的ThreadingMiXin来给其他网络相关类增加多线程支持
from xmlrpc.server import SimpleXMLRPCServer
from socketserver import ThreadingMixIn
class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass
# 关于混入类，有几点需要记住
# 首先：混入类不能直接被实例化使用
# 其次：混入类没有自己的状态信息，它们没有定义__init__方法，并且没有实例属性，这也是伤心买呢为何定义了__slots__=()

# 还有一种实现混入类的方式是使用类装饰器
# 使用装饰器，给原有类的相关的方法扩展了日志功能
def LoggedMapping(cls):
    '''第二种实现混入类的方式：使用装饰器'''
    cls_getitem = cls.__getitem__
    cls_setitem = cls.__setitem__
    cls_delitem = cls.__delitem__

    def __getitem__(self, key):
        print('Getting: ' + str(key))
        return cls_getitem(self, key)
    
    def __setitem__(self, key, value):
        print('Setting {} = {!r}'.format(key, value))
        return cls_setitem(self, key)
    
    def __delitem__(self, key):
        print('Deleting ' + str(key))
        return cls_delitem(self, key)
    cls.__getitem__ = __getitem__
    cls.__setitem__ = __setitem__
    cls.__delitem__ = __delitem__
    return cls

@LoggedMapping
class LoggedDict(dict):
    pass

