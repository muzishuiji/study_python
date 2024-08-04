from collections import OrderedDict
import collections
from typing import Any, MutableMapping

# a set of descriptors for various types
class Typed:
    _expected_type = type(None)
    def __init__(self, name=None):
        self._name = name
    
    def __set__(self, instance, value):
        if not isinstance(value, self._expected_type):
            raise TypeError('Expected ' + str(self._expected_type))
        instance.__dict__[self._name] = value
    
class Integer(Typed):
    _expected_type = int

class Float(Typed):
    _expected_type = float

class String(Typed):
    _expected_type = str

# metaclass that uses an orderedDict for class body
class OrderedMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        d = dict(clsdict)
        order = []
        for name, value in clsdict.items():
            if isinstance(value, Typed):
                value._name = name
                order.append(name)
        d['_order'] = order
        return type.__new__(cls, clsname, bases, d)
    @classmethod
    def __prepare__(cls, clsname, bases):
        # OrderedDict可以很容易的捕获定义的顺序
        return OrderedDict()
    
class Structure(metaclass=OrderedMeta):
    def as_csv(self):
        # 将一个类实例的数据序列化为一行csv数据
        return ','.join(str(getattr(self, name)) for name in self._order)
    
# example use
class Stock(Structure):
    name = String()
    shares = Integer()
    price = Float()

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

s = Stock('GOOG',100,490.1)
s.as_csv()
# 'GOOG,100,490.1'
t = Stock('AAPL','a lot', 610.23)
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "dupmethod.py", line 34, in __init__
# TypeError: shares expects <class 'int'>

# 构造自己的类字典对象，下面的方案可以防止重复定义
from collections import OrderedDict
class NoDupOrderedDict(OrderedDict):
    def __init__(self, clsname):
        self.clsname = clsname
        super().__init__()
    def __setitem__(self, name, value):
        if name in self:
            raise TypeError('{} already defined in {}'.format(name, self.clsname))
        super().__setitem__(name, value)
    
class OrderedMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        d = dict(clsdict)
        d['_order'] = [name for name in clsdict if name[0] != '_']
        return type.__new__(cls, clsname, bases, d)

    @classmethod
    def __prepare__(cls, clsname, bases):
        return NoDupOrderedDict(clsname)

# 能够捕获类定义的顺序是一个看似不起眼却又非常重要的特性，可以很方便的捕获定义的顺序来将对象映射到元组或数据表中的行
# python允许我们使用metaclass关键字参数来指定特定的元类，如抽象基类
from abc import ABCMeta, abstractmethod
class IStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxsize=None):
        pass
    @abstractmethod
    def write(self, data):
        pass
# 在自定义元类中我们还可以提供其他的关键字参数
class Spam(metaclass=MyMeta, debug=True, synchronize=True):
    pass

# check下init和call的执行时机
class MyClass:
    def __init__(self):
        print("Initializing instance.")
    def __call__(self, *args, **kwds) :
        print("MyClass called.")


obj = MyClass()  # 输出: "Initializing instance."
obj()


# python允许我们使用metaclass关键字参数来指定特定的元类
from abc import ABCMeta, abstractmethod
class IStream(mateclass=ABCMeta):
    @abstractmethod
    def read(self, maxsize=None):
        pass
    @abstractmethod
    def write(self, data):
        pass

# 在自定义元类中提供其他的关键字参数
class Spam(metaclass=MyMeta, debug=True, synchronize=True):
    pass
# 为了使元类支持这些关键字参数，你需要确保在__prepare__(),__new__(), __init__方法中使用强制关键字参数
class MyMeta(type):
    # Optional
    # 会在所有类主体执行前被执行
    @classmethod
    def __prepare__(cls, name, bases, *, debug=False, synchronize=False):
        # custom processing
        pass
        return super().__prepare__(name, bases)
    
    # Required
    # new方法在类的主体被执行完后开始执行，用来实例化最终的类对象
    def __new__(cls, name, bases, ns, *, debug=False, synchronize=False):
        # custom processing
        pass
        return super().__new__(cls, name, bases, ns)
    
    # required
    # init方法最后被调用，用来执行其他的初始化工作
    def __init__(self, name, bases, ns, *, debug=False, synchronize=False):
        # custom processing
        pass
        return super().__init__(name, bases, ns)
    
# 关键字参数配置一个元类可以视作对类变量的一种替代方式
class Spam(metaclass=MyMeta):
    debug = True
    synchronize = True
    pass


# 强制函数签名，基类中定义一个非常通用的init方法，然后强制所有的子类必须提供一个特定的参数签名
from inspect import Signature, Parameter
# 生成签名
def make_sig(*names):
    parms = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in names]
    return Signature(parms)

class Structure:
    __signature__ = make_sig()
    def __init__(self, *args, **kwargs):
        bound_values = self.__signature__.bind(*args, **kwargs)
        for name, value in bound_values.arguments.items():
            setattr(self, name, value)
# example use
class Stock(Structure):
    __signature__ = make_sig('name', 'shares', 'price')

class Point(Structure):
    __signature__ = make_sig('x', 'y')

import inspect
print(inspect.signature(Stock))
# (name, shares, price)
s1 = Stock('ACME', 100, 490.1)
s2 = Stock('ACME', 100)
# Traceback (most recent call last):
# ...
# TypeError: 'price' parameter lacking default value
s3 = Stock('ACME', 100, 490.1, shares=50)
# Traceback (most recent call last):
# ...
# TypeError: multiple values for argument 'shares'


# 使用自定义元类来创建签名对象
from inspect import Signature, Parameter
def make_sig(*names):
    parms = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in names]
    return Signature(parms)

class StructureMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        # *clsdict.get('_fields', []) 创建fields的key的元组
        clsdict['__signature__'] = make_sig(*clsdict.get('_fields', []))
        return super().__new__(cls, clsname, bases, clsdict)
    
class Structure(metaclass=StructureMeta):
    _fields=[]
    def __init__(self, *args, **kwargs):
        bound_values = self.__signature__.bind(*args, **kwargs)
        for name, value in bound_values.arguments.items():
            setattr(self, name, value)

# example use
class Stock(Structure):
    _fields = ['name', 'shares', 'price']

class Point(Structure):
    _fields = ['x', 'y']

# 自定义签名的时候吧，将签名存储在特定的属性__signature__ 中通常是很有用的，这样的话，在使用inspect模块执行内省的代码就能发现签名并将它作为调用约定
import inspect
print(inspect.signature(Stock))
# (name, shares, price)
print(inspect.signature(Point))
# (x, y)


# 如果你想监控类的定义，可以定义一个元类
# 一个基本元类通常是继承自type并重定义它的__new__方法或__init__方法
class MyMeta(type):
    def __new__(self, clsname, bases, clsdict):
        # clsname is name of class being defined
        # bases is tuple of base classes
        # clsdict is class dictionary
        return super().__new__(cls, clsname, bases, clsdict)

class MyMeta(type):
    def __init__(self, clsname, bases, clsdict):
        super().__init__(clsname, bases, clsdict)
        # clsname is name of class being defined
        # bases is tuple of base classes
        # clsdict is class dictionary

# 顶级父类
class Root(metaclass=MyMeta):
    pass
class A(Root):
    pass
class B(Root):
    pass

# 一个框架的构建者就能在大型的继承体系中通过一个顶级父类制定一个元类去捕获所有下面子类的定义
# 定义一个元类约束不可用驼峰命名method
from inspect import signature
import logging

class MatchSignatureMeta(type):
    def __init__(self, clsname, bases, clsdict):
        for name in clsdict:
            if name.lower() != name:
                raise TypeError('Bad attribute name: ' + name)
        return super().__new__(cls, clsname, bases, clsdict)
                    
# example
class Root(metaclass=MatchSignatureMeta):
    pass

class A(Root):
    def foo_bar(self): # ok
        pass
class B(Root):
    def fooBar(self): # TypeError
        pass

# 检测重载方法，确保它的调用参数跟父类中原始方法有着相同的参数签名
from inspect import signature
import logging

class MatchSignatureMeta(type):
    def __init__(self, clsname, bases, clsdict):
        super().__init__(clsname, bases, clsdict)
        # 用于寻找位于继承体系中构建self父类的定义
        sup = super(self, self)
        for name, value in clsdict.items():
            if name.startswith('_') or not callable(value):
                continue
            # get the previous definition if any and compare the signatures
            # 获取父类的参数元组？？？
            prev_dfn = getattr(sup, name, None)
            if prev_dfn:
                prev_sig = signature(prev_dfn)
                val_sig = signature(value)
                if prev_sig != val_sig:
                    logging.warning('Signature mismatch in %s. %s != %s',
                                    value.__qualname__, prev_sig, val_sig)
# example
class Root(metaclass=MatchSignatureMeta):
    pass

class A(Root):
    def foo(self, x, y):
        pass
    def spam(self, x, *, y):
        pass
# class with redefined methods,but slightly different signatures
class B(A):
    def foo(self, x, y):
        pass
    def spam(self, x, z):
        pass

# 在大型面向对象的程序中，将类的定义放在元类中控制是有用的
# 元类可以监控类的定义，警告编程人员没有注意到的可能出现的问题
# 通常用于通过某种方式（比如通过改变类字典的内容）修改类的定义
# new 方法是在类创建之前被调用，init方法在类创建之后被调用
# 当需要构建类对象的时候会有用，如需要使用super函数探索之前的定义，则只能在实例创建之后
# 很多时候如果能够构造新的类对象是很有用的

# 创建一个新的类对象，将类的源代码以字符串的形式发不出去，并使用函数exec来执行它

# stock.py
# example of making a class manually from parts
# methods
def __init__(self, name, shares, price):
    self.name = name
    self.shares = shares
    self.price = price

def cost(self):
    return self.shares * self.price

cls_dict = {
    '__init__': __init__,
    'cost': cost
}

# make a class
import types
# 用不同的基础函数构建出一个散称的类？？？，底层用exec来创建的？？？
Stock = types.new_class('Stock', (), {}, lambda ns: ns.update(cls_dict))
Stock.__module__ = __name__
print(Stock)
# <class '__main__.Stock'>

# example use
s = Stock('ACME', 50, 91.1)
s.cost()
# 4555.0

# 创建的类需要一个不同的元类
import abc
Stock = types.new_class('Stock', (), {'metaclass': abc.ABCMeta}, lambda ns: ns.update(cls_dict))
Stock.__module__ = __name__
print(Stock)
# <class '__main__.Stock'>
type(Stock)
# <class 'abc.ABCMeta'>

# 定义一个包含关键字参数的类
Spam = types.new_class('Spam', (Base,), {'debug': True, 'typecheck': False}, lambda ns: ns.update(cls_dict))

# 很多时候能够构造新的类对象是有用的,namedtuple使用exec() 创建
Stock = collections.namedtuple('Stock', ['name', 'shares', 'price'])
Stock # <class '__main__.Stock'>

import operator
import types
import sys

def named_tuple(classname, fieldnames):
    # populate a dictionary of field property
    # 填充一个字段属性访问的字典
    cls_dict = {name: property(operator.itemgetter(n)) for n, name in enumerate(fieldnames)}
    # make a __new__ function add to class dict
    def __new__(cls, *args):
        if len(args) != len(fieldnames):
            raise TypeError('Excepted {} arguments'.format(len(fieldnames)))
        return tuple.__new__(cls, args)
    # 包装创建类的new方法
    cls_dict['__new__'] = __new__
    # make the class
    cls = types.new_class(classname, (tuple,), {}, lambda ns: ns.update(cls_dict))
    # 将模块设置为调用者的模块
    # 框架魔法，sys._getframe 来获取调用者的模块名
    cls.__module__ = sys._getframe(1).f_globals['__name__']
    return cls

Point = named_tuple('Point', ['x', 'y'])
# <class '__main__.Point'>
p = Point(4,5)
len(p) # 2
p.x = 2
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
# AttributeError: can't set attribute

# 你可以通过直接实例化一个元类来直接创建一个类
Stock = type('Stock', (), cls_dict)
# 上面方法的问题是会忽略一些关键步骤，如元类中__prepare__方法的调用
# 使用types.new_class可以保证所有必要的初始化步骤得到执行，type.new_class()第四个参数的回调函数接收__prepare__返回的映射对象
# 如果只想执行准备步骤，使用type.prepare_class()
import types
metaclass, kwargs, ns = type.prepare_class('Stock', (), {'metaclass': type})
# 会找到合适的元类并调用它的__prepare__方法，然后这个元类保存它的关键字参数准备命名空间后被返回
# new方法在实例创建之前被触发，init方法在实例创建之后被触发

# 在元类中定义一些继承它的类的一些初始化规则
import operator

class StructTupleMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 修改元组的调用签名，元组一旦创建就不可修改
        for n, name in enumerate(cls._fields):
            # operator.itemgetter 创建访问器函数，property函数将其转换成一个属性
            setattr(cls, name, property(operator.itemgetter(n)))

class StructTuple(tuple, metaclass=StructTupleMeta):
    _fields = []
    # 实例创建之前被触发，前置的拦截不符合规范的实例化调用
    def __new__(cls, *args):
        if len(args) != len(cls._fields):
            raise ValueError('{} arguments required, but got {}'.format(len(cls._fields), len(args)))
        return super().__new__(cls, args)
    
class Stock(StructTuple):
    _fields = ['name', 'shares', 'price']

class Point(StructTuple):
    _fields = ['x', 'y']

s = Stock('ACME', 50, 91.1, 2)
s = Stock('ACME', 50, 91.1)
s.shares = 1
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: can't set attribute

# python中允许通过函数注解实现重载
class Spam:
    def bar(self, x: int, y: int):
        print('Bar 1:', x, y)
    
    def bar(self, s: str, n: int = 0):
        print('Bar 2:', s, n)
    
s = Spam()
s.bar(2, 3) # Prints Bar 1: 2 3
s.bar('hello') # Prints Bar 2: hello 0

# 借助于函数注解实现方法重载
import inspect
import types

class MultiMethod:
    '''
    Represents a single multimethod.
    '''
    def __init__(self, name):
        self._methods = {}
        self.__name__ = name
    
    def register(self, meth):
        '''
        Register a new method as a multimethod
        '''
        sig = inspect.signature(meth)
        # build a type signature from the method's annotations
        types = []
        for name, parm in sig.parameters.items():
            if name == 'self':
                continue
            if parm.annotation is inspect.Parameter.empty:
                raise TypeError(
                    'Argument {} must be annotated with a type'.format(name)
                )
            # 注解非合法type
            if not isinstance(parm.annotation, type):
                raise TypeError(
                    'Argument {} annotation must be a type'.format(name)
                )
            # 有默认值的也记为一种类型的method
            if parm.default is not inspect.Parameter.empty:
                self._methods[tuple(types)] = meth
            types.append(parm.annotation)
        self._methods[tuple(types)] = meth
    # call方法从所有排出self的参数中构建一个类型元组，在内部map中查找这个方法，然后调用相应的方法
    def __call__(self, *args):
        '''
        Call a method bases on type signature of the arguments
        '''
        types = tuple(type(arg) for arg in args[1:])
        meth = self._methods.get(types, None)
        if meth:
            return meth(*args)
        else:
            raise TypeError('No matching method for types {}'.format(types))
    # 用来构建正确的绑定方法    
    def __get__(self, instance, cls):
        '''
        Descriptor method needed to make calls work in a class
        需要描述符方法才能使类中的调用起作用
        '''
        if instance is not None:
            return type.MethodType(self, instance)
        else:
            return self
        
class MultiDict(dict):
    '''
    Special dictionary to build multimethods in a metaclass
    '''
    def __setitem__(self, key, value):
        if key in self:
            # if key already exists,it must be a multimethod or callable
            current_value = self[key]
            if isinstance(current_value, MultiMethod):
                current_value.register(value)
            else:
                mvvalue = MultiMethod(key)
                mvvalue.register(current_value)
                mvvalue.register(value)
                super().__setitem__(key, mvvalue)
        else:
            super().__setitem__(key, value)
class MultipleMeta(type):
    '''
    Metaclass that allows multiple dispatch of methods
    '''
    def __new__(cls, clsname, bases, clsdict):
        return type.__new__(cls, clsname, bases, dict(clsdict))
    @classmethod
    def __prepare__(cls, clsname, bases):
        return MultiDict()
    
class Spam(metaclass=MultipleMeta):
    def bar(self, x: int, y:int):
        print('Bar 1:', x, y)
    
    def bar(self, x: str, y:int = 0):
        print('Bar 1:', x, y)

# Example: overloaded __init__
import time
class Date(metaclass=MultipleMeta):
    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day
    
    def __init__(self):
        t = time.localtime()
        self.__init__(t.tm_year, t.tm_mon, t.tm_day)

s = Spam()
s.bar(2, 3)
# Bar 1: 2 3
s.bar('hello')
# Bar 2: hello 0
s.bar('hello', 5)
# Bar 2: hello 5
s.bar(2, 'hello')
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "multiple.py", line 42, in __call__
#         raise TypeError('No matching method for types {}'.format(types)

d = Date(2012, 12, 21)
# get today's date
e = Date()
e.year
# 2012
e.month
# 12
e.day
# 3
# 对函数注解实现函数重载有两个问题：
# -不能使用关键字参数
# -对继承也是有限制的
# 使用元类和注解的一种替代方案，可以通过描述器来实现类似的效果
import types
class multimethod:
    def __init__(self, func):
        self._methods = {}
        self.__name__ = func.__name__
        self.default = func
    def match(self, *types):
        def register(func):
            ndefaults = len(func.__defaults__) if func.__defaults__ else 0
            for n in range(ndefaults + 1):
                self._methods[types[:len(types) - n]] = func
            return self
        return register
    
    def __call__(self, *args):
        types = tuple(type(arg) for arg in args[1:])
        meth = self._methods.get(types, None)
        if meth:
            return meth(*args)
        else:
            return self._default(*args)
    
    def __get__(self, instance, cls):
        if instance is not None:
            return types.MethodType(self, instance)
        else:
            return self

class Spam:
    @multimethod
    def bar(self, *args):
        # default method called if no match
        raise TypeError('No matching method for bar')
    @bar.match(int, int)
    def bar(self, x, y):
        print('Bar 1:', x, y)
    @bar.match(str, int)
    def bar(self, s, n = 0):
        print('Bar 1:', s, n)  

