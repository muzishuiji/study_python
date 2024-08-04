# 定义一个统计耗时的装饰器函数
import time
from functools import wraps

def timethis(func):
    '''
    Decorator that reports the execution time
    '''
    # 这个注解是很重要的，它能保留原始函数的元数据
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        # 装饰器并不改变原来的返回值
        return result
    return wrapper

# 使用装饰器的例子
@timethis
def countdown(n):
    '''
    Counts down
    '''
    while n > 0:
        n -= 1

countdown(100000)
# countdown 0.008917808532714844
countdown(10000000)
# countdown 0.87188299392912

# 使用装饰器的方法之一是使用注解
# 使用装饰器的方法之二是直接调用
def countdown(n):
    pass
countdown = timethis(countdown)


# 在你任何时候定义装饰器的时候，都应该使用functools库中的@wraps装饰器来注解底层包装函数
countdown.__name__
# 'countdown'
countdown.__doc__
# '\n\tCounts down\n\t'
countdown.__annotations__
# {'n': <class 'int'>}
# 如果你忘记使用@wraps，你会发现被装饰函数丢失了所有有用信息
# @wraps有一个重要特征是它能让你通过属性__wrapped__直接访问包装函数
countdown.__wrapped__(100000)
# __wrapped__ 属性还能让装饰器函数正确暴露底层的参数签名信息
from inspect import signature
print(signature(countdown))
# (n:int)

# 假设装饰器是通过@wraps来实现的，你可以通过访问__wrapped__属性来访问原始函数
@somedecorator
def add(x, y):
    return x + y

orig_add = add.__wrapped__

orig_add(3, 4) # 7
# 如果有多个包装器，访问__wrapped__属性的行为是不可预知的，在python3.3中，会略过所有的包装层
from functools import wraps

def decorator1(func):
    @wraps
    def wrapper(*args, **kwargs):
        print('Decorator 1')
        return func(*args, **kwargs)
    return wrapper


def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('Decorator 2')
        return func(*args, **kwargs)
    return wrapper

@decorator1
@decorator2
def add(x, y):
    return x + y

add(2, 3)
# Decorator 1
# Decorator 2
# 5
add.__wrapped__(2, 3) # 5
# 并不是所有的装饰器都使用了wraps，内置的装饰器@staticmethod，@classmethod就没有遵循这个约定
#（它们把原始函数属性存储在__func__中）
# 定义一个可以接受参数的装饰器
from functools import wraps
import logging

def logged(level, name=None, message=None):
    '''
    Add logging to a function. level is the logging
    level, name is the logger name, and message is the
    log message. If name and message aren't specified,
    they default to the function's module and name.
    '''
    def decorate(func):
        # 将装饰器传入的参数提取出来
        logname = name if name else func.__module__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)
        return wrapper
    return decorate


# example use
logged(logging.DEBUG)
def add(x, y):
    return x + y

@logged(logging.CRITICAL, 'example')
def spam():
    print('Spam!')

# 例如下面这样的代码
@decorator(x, y, z)
def func(a, b):
    pass

# 装饰器的处理过程跟下面的调用是等效的
def func(a, b):
    pass
func = decorator(x, y, z)(func)
# decorator(x, y, z) 的返回结果必须是一个可调用对象，它接受一个函数作为参数并包装它

# 通过引入一个访问函数，使用nonlocal来修改内部变量，然后这个访问函数被作为一个属性赋值给包装函数吧
from functools import wraps, partial
import logging
# utility decorator to attach a function as an attribute of obj
# 装饰器是一种特殊的工具，用于将函数附加为对象的属性
def attach_wrapper(obj, func=None):
    if func is None:
        # 返回一个未完全初始化的自身，除了被包装函数外其他的参数都已经确定袭来
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func

def logged(level, name=None, message=None):
    '''
    Add logging to a function. level is the logging
    level, name is the logger name, and message is the
    log message. If name and message aren't specified,
    they default to the function's module and name.
    '''
    def decorate(func):
        logname = name if name else func.__module__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)
        # attach setter functions
        @attach_wrapper(wrapper)
        def set_level(newlevel):
            nonlocal level
            level = newlevel

        @attach_wrapper(wrapper)
        def set_message(newmsg):
            nonlocal logmsg
            logmsg = newmsg

        return wrapper
    return decorate
# example use
@logged(logging.DEBUG)
def add(x, y):
    return x + y

@logged(logging.CRITICAL, 'example')
def spam():
    print('Spam！')

import logging
logging.basicConfig(level=logging.DEBUG)
add(2, 3)
# DEBUG:__main__:add
# change the log message
add.set_message('Add called')
add(2, 3)
# DEBUG:__main__:Add called
# 5
# change the log level
add.set_level(logging.WARNING)
add(2, 3)
# WARNING:__main__:Add called
# 5

# 一个比较难理解的地方就是对于访问函数的首次使用，你可能会考虑另外一个方法直接访问函数的属性
@wraps(func)
def wrapper(*args, **kwargs):
    wrapper.log.log(wrapper.level, wrapper.logmsg)
    return func(*args, **kwargs)

# attach adjustable attributes
wrapper.level = level
wrapper.logmsg = logmsg
wrapper.log = log
# 但上述实现要求它必须是最外层的装饰器才行，如果不是，它会隐藏底层属性，使得修改它们没有任何作用
# 而通过访问函数就能避免这样的局限性
# 定义一个参数是可选的装饰器
from functools import wraps, partial
import logging

def logged(func=None, *, level=logging.DEBUG, name=None,message=None):
    if func is None:
        return partial(logged, level=level, name=name, message=message)
    logname = name if name else func.__module__
    log = logging.getLogger(logname)
    logmsg = message if message else func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        log.log(level, logmsg)
        return func(*args, **kwargs)
    return wrapper

# example use
@logged
def add(x, y):
    return x + y

@logged(level=logging.CRITICAL, name='example')
def spam():
    print('Spam~')
# 如果有参数被传递进来，装饰器要返回一个接受一个函数参数并包装它的函数


class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 == 1:
            return False
        pairs = {
            ')': '(',
            ']': '[',
            '}': '{'
        }
        stack = list()
        for ch in s:
            if ch in pairs:
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
            else:
                stack.append(ch)
        return not stack

# 对参数做强制类型检查
@typeassert(int, int)
def add(x, y):
    return x + y

add(2, 3)
add(2, 'hello')
# 用装饰器技术来实现@typeassert
from inspect import signature
from functools import wraps

def typeassert(*ty_args, **ty_kwargs):
    def decorate(func):
        # if in optimized mode, disable type checking
        if not __debug__:
            return func
        # map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            'Argument {} must be {}'.format(name, bound_types[name])
                            )
            return func(*args, **kwargs)
        return wrapper
    return decorate

@typeassert(int, list)
def bar(x, items=None):
     if items is None:
         items = []
     items.append(x)
     return items

# 文档里说对于有默认参数的typeassert并不适用，我感觉不是这样，嗯，我是对的
bar(2)
# [2]
bar(2,'2')
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "contract.py", line 33, in wrapper
# TypeError: Argument items must be <class 'list'>

# 上述装饰器非常灵活，既可以指定所有参数类型，也可以指定部分
@typeassert(int, z=int)
def spam(x, y, z = 42):
    print(x,y,z)


spam(1, 'hello', 3)
# 1 hello 3
spam(1, 'hello', 'world')
# Traceback (most recent call last):
# File "<stdin>", line 1, in <module>
# File "contract.py", line 33, in wrapper
# TypeError: Argument z must be <class 'int'>

# 可以通过制定参数，去掉装饰器的功能，只简单返回被装饰函数
# 可以使用inspect.signature()函数来获取可调用对象的签名信息
from inspect import signature
def spam(x, y, z = 42):
    pass

sig = signature(spam)
print(sig)
# (x, y, z=42)
sig.parameters['z'].name
# 'z'
sig.parameters['z'].default
# 42
sig.parameters['z'].kind
# <_ParameterKind: 'POSITIONAL_OR_KEYWORD'>
# bind_partial() 方法获取指定类型到名称的部分绑定
# bind和bind_partial的去背诗，它不会忽略任何参数

# 在类中定义装饰器
# @property装饰器实际上是一个类，它定义了三个方法getter,setter,deleter,每一个方法都是一个装饰器
class Person:
    # create a property instance
    first_name = property()

    # apply decorator methods
    @first_name.getter
    def first_name(self):
        return self._first_name
    
    @first_name.setter
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('expected a string')
        self._first_name = value
# 这样定义的主要原因是各种不同的装饰器方法会在关联的property实例上操作它的状态
# 任何时候只要你碰到需要在装饰器中记录或绑定信息，都可以这样做

from functools import wraps

class A:
    # decorator as an instance method
    def decorator1(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print('Decorator 1')
            return func(*args, **kwargs)
        return wrapper
    
    # decorator as a class method
    @classmethod
    def decorator2(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print('Decorator 2')
            return func(*args, **kwargs)
        return wrapper
# example
a = A()
@a.decorator1
def spam():
    pass

# as a class method
@A.decorator2
def grok():
    pass

class B(A):
    # 引用定义在类方法上的装饰器
    @A.decorator2
    def bar(self):
        pass

b = B()
b.bar()
@B.decorator2
def grokB():
    pass


# 如果想要实现一个可以记录和修改状态的装饰器有两种方案：
# 方案一：将装饰器定义为类，使用类属性保存状态
# 装饰器定义成一个实例，需要确保实现了__call__()和__get__()方法
import types
from functools import wraps

class Profiled:
    def __init__(self) -> None:
        wraps(func)(self)
        self.ncalls = 0
    
    def __call__(self, *args, **kwds):
        self.ncalls += 1
        return self.__wrapped__(*args, **kwds)
    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)
@Profiled
def add(x, y):
    return x + y

class Spam:
    @Profiled
    def bar(self, x):
        print(self, x)
add(2, 3)
add(2, 3)
add.ncalls # 2
s = Spam()
s.bar(1)
s.bar(2)
s.bar(3)
Spam.bar.ncalls #3
# 方案二：使用闭包和nonlocal变量来实现装饰器，创建一个为函数的属性来保存状态
import types
from functools import wraps

def profiled(func):
    ncalls = 0
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal ncalls
        ncalls += 1
        return func(*args, **kwargs)
    # 包装的结果就是在被包装对象上挂了一个属性，属性是一个方法，方法的逻辑是返回ncalls变量
    wrapper.ncalls = lambda: ncalls
    return wrapper
# example 
@profiled
def add(x, y):
    return x + y
add(2, 3)
add(4, 5)
add.ncalls()


# 装饰器的执行顺序是从内到外
# 给类或静态方法提供装饰器是很简单的，不过要确保装饰器在@classmethod或@staticmethod之前
# 问题:@classmethod和@staticmethod实际上并不会创建可直接调用的对象，而是创建特殊的描述器对象，
# 取保这种装饰器出现在装饰器链中第一个位置就不会出错
import time
from functools import wraps

# a simple decorator
# 一个拥有同级函数执行时长的装饰器
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        print(end - start)
        return r
    return wrapper
# Class illustrating application of the decorator to different kinds of methods
class Spam:
    @timethis
    def instance_method(self, n):
        print(self, n)
        while n > 0:
            n -= 1

    @classmethod
    @timethis
    def class_method(cls, n):
        print(cls, n)
        while n > 0:
            n -= 1
    @staticmethod
    @timethis
    def static_method(cls, n):
        print(cls, n)
        while n > 0:
            n -= 1

s = Spam()
s.instance_method(1000000)
# <__main__.Spam object at 0x1006a6050> 1000000
# 0.11817407608032227
Spam.class_method(1000000)
# <class '__main__.Spam'> 1000000
# 0.11334395408630371
Spam.static_method(1000000)
# 1000000
# 0.11740279197692871

# 抽象基类中定义类和抽象方法，注意装饰器的顺序
from abc import ABCMeta, abstractmethod
class A(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def method(cls):
        pass

# 使用关键字参数来给包装函数增加参数
from functools import wraps

def optional_debug(func):
    @wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if debug:
            print('Calling', func.__name__)
        return func(*args, **kwargs)
    return wrapper

@optional_debug
def spam(a, b, c):
    print(a, b, c)

spam(1,2,3) # 1 2 3

spam(1,2,3, debug=True) 
# Calling spam
# 1 2 3

# 其实关键不是加参数，而是借助于参数和相关的功能逻辑给被包装的函数添加一些通用逻辑，以避免重复代码
def a(x, debug=False):
    if debug:
        print('Calling a')
def b(x,y,z, debug=False):
    if debug:
        print('Calling a')
def c(x,y, debug=False):
    if debug:
        print('Calling c')  
    
# 重构，增加一些校验逻辑
from functools import wraps
import inspect

def optional_debug(func):
    if 'debug' in inspect.getargspec(func).args:
        raise TypeError('debug argument already defined')
    @wraps(func)
    def wrapper(*args, debug=False,**kwargs):
        if debug:
            print('Calling', func.__name__)
        return func(*args, **kwargs)
    return wrapper

@optional_debug
def a(x):
    pass
@optional_debug
def b(x,y,z):
    pass

@optional_debug
def c(x, y):
    pass

# 强制关键字参数很容易被添加到接受*args，**kwargs参数的函数中
# 调整解决被包装的函数签名不符合预期的问题
from functools import wraps
import inspect

def optional_debug(func):
    if 'debug' in inspect.getargspec(func).args:
        raise TypeError('debug argument already defined')

    @wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if debug:
            print('Calling', func.__name__)
        return func(*args, **kwargs)
    
    # 处理函数签名
    sig = inspect.signature(func)
    parms = list(sig.parameters.values())
    parms.append(inspect.Parameter('debug', inspect.Parameter.KEYWORD_ONLY, default=False))
    wrapper.__signature__ = sig.replace(parameters=parms)
    return wrapper
# 修改后，包装后的签名能正确的显示debug参数的存在
@optional_debug
def add(x, y):
    return x + y

print(inspect.signature(add))
# (x, y, *, debug=False)

# 1. func 定义装饰在类上的装饰器
# 一个重写了特殊方法__getattribute__的类装饰器
def log_getattribute(cls):
    # get the original implementation
    orig_getattribute = cls.__getattribute__

    # make a new definition
    def new_getattribute(self, name):
        print('getting:', name)
        return orig_getattribute(self, name)
    # attach to the class and return
    cls.__getattribute__ = new_getattribute
    return cls
# example use
@log_getattribute
class A:
    def __init__(self, x):
        self.x = x
    def spam(self):
        pass
a = A(42)
a.x
# getting: x
# 42
a.spam()
# getting: spam

# 2.  定义装饰在类上的装饰器
# 借助于继承来完成对类的某些方法或者能力的混入
class LoggedGetattribute:
    def __getattribute__(self, name: str):
        print('getting:', name)
        return super().__getattribute__(name)
# example
class A(LoggedGetattribute):
    def __init__(self, x):
        self.x = x
    def spam(self):
        pass

#  如果你要对类使用多个装饰器，需要注意放置顺序，避免产生不符合预期的结果

# 可以通过自定义__call__方法来实现自定义创建实例的逻辑
# 禁止创建类的实例
class NoInstances(type):
    def __call__(self, *args, **kwds):
        raise TypeError("Can't instantiate directly")
    
# example
class Spam(metaclass=NoInstances):
    @staticmethod
    def grok(x):
        print('Spam.grok')

# 实现单例模式
class Singleton(type):
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)
    
    def __call__(self, *args, **kwds):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwds)
            return self.__instance
        else:
            return self.__instance
# example
class Spam(metaclass=Singleton):
    def __init__(self):
        print('Creating Spam')

# 如果想要创建缓存实例，可以通过元类实现
import weakref
class Cached(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__cache = weakref.WeakValueDictionary()
    def __call__(self, *args, **kwds):
        if args in self.__cache:
            return self.__cache[args]
        else:
            obj = super().__call__(*args)
            self.__cache[args] = obj
            return obj
# example
class Spam(metaclass=Cached):
    def __init__(self, name):
        print('Creating Spam({!r})'.format(name))
        self.name = name
a = Spam('Guido')
# Creating Spam('Guido')
b = Spam('Diana')
# Creating Spam('Diana')
c = Spam('Guido') 
a is c # True

# 如果你想通过工厂函数实现单例模式，你需要将类隐藏在某些工厂函数后面
class _Spam:
    def __init__(self):
        print('Creating Spam')
_spam_instance = None
def Spam():
    global _spam_instance

    if _spam_instance is not None:
        return _spam_instance
    else:
        _spam_instance = _Spam()
        return _spam_instance
