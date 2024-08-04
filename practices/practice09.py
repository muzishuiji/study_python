import numpy as np
import matplotlib.pyplot as plt

# 假设有以下数据点
x = np.array([20, 50, 80, 130, 210, 280])
y = np.array([55, 65, 80, 95, 80, 50])

# 使用numpy的polyfit进行曲线拟合，这里选择了二次多项式拟合
p = np.polyfit(x, y, 2)

# 得到拟合之后的多项式函数
f = np.poly1d(p)

# 生成新的x值
x_new= np.linspace(x[0], x[-1], 500)

# 计算对应的y值
y_new = f(x_new)

# 画出原始数据点和拟合曲线
plt.scatter(x, y, color='b') # 原始数据点
plt.plot(x_new, y_new, color='r') # 拟合曲线

plt.show()

# 打印出拟合方程
print(f)

# 使用装饰器functools.total_ordering 可以简化让类支持比较操作的问题，只需要定义一个eq方法，外加其他方法__lt__,__le__, __gt__,__ge__中的一个即可
# 然后装饰器会自动为你填充其他比较方法
from functools import total_ordering
class Room:
    def __init__(self, name, length, width):
        self.name = name
        self.length = length
        self.width = width
        self.square_feet = self.length * self.width

@total_ordering
class House:
    def __init__(self, name, style):
        self.name = name
        self.style = style
        self.rooms = list()
    
    @property
    def living_space_footage(self):
        return sum(r.square_feet for r in self.rooms)
    
    def add_room(self, room):
        return self.rooms.append(room)
    
    def __str__(self):
        return '{}: {} square foot {}'.format(self.name,
                                              self.living_space_footage,
                                              self.style)
    def __eq__(self, other):
        return self.living_space_footage == other.living_space_footage

    def __lt__(self, other):
        return self.living_space_footage < other.living_space_footage
    
# Build a few houses, and add rooms to them
h1 = House('h1', 'Cape')
h1.add_room(Room('Master Bedroom', 14, 21))
h1.add_room(Room('Living Room', 18, 20))
h1.add_room(Room('Kitchen', 12, 16))
h1.add_room(Room('Office', 12, 12))
h2 = House('h2', 'Ranch')
h2.add_room(Room('Master Bedroom', 14, 21))
h2.add_room(Room('Living Room', 18, 20))
h2.add_room(Room('Kitchen', 12, 16))
h3 = House('h3', 'Split')
h3.add_room(Room('Master Bedroom', 14, 21))
h3.add_room(Room('Living Room', 18, 20))
h3.add_room(Room('Office', 12, 16))
h3.add_room(Room('Kitchen', 15, 17))
houses = [h1, h2, h3]
print('Is h1 bigger than h2?', h1 > h2) # prints True
print('Is h2 smaller than h3?', h2 < h3) # prints True
print('Is h2 greater than or equal to h1?', h2 >= h1) # Prints False
print('Which one is biggest?', max(houses)) # Prints 'h3: 1101-square-foot Split'
print('Which is smallest?', min(houses)) # Prints 'h2: 846-square-foot Ranch'

# total_ordering装饰器其实只是定义了一个从每个比较支持方法到所有需要定义的其他方法的一个映射而已
# 比如你定义了le方法，它就被用来构建所有其他的需要定义的那些特殊方法
class House:
    def __eq__(self, other):
        pass
    def __lt(self, other):
        pass
    # methods created by @total_ordering
    __lt__ = lambda self, other: self < other or self == other
    __gt__ = lambda self, other: not (self < other or self == other)
    __ge__ = lambda self, other: not (self < other)
    __ne__ = lambda self, other: not self == other

# 创建一个类的对象时，如果之前使用同样参数创建过这个对象，你想返回它的缓存引用
# 相同参数创建的对象是单例的，可以使用logging模块
import logging
a = logging.getLogger('foo')
b = logging.getLogger('bar')
a is b
# False
c = logging.getLogger('foo')
a is c
# True

# 为了达到这样的效果，你需要使用一个和类本身分开的工厂函数
# the class in question
class Spam:
    def __init__(self, name):
        self.name = name

# Caching support
import weakref
_spam_cache = weakref.WeakKeyDictionary()
def get_spam(name):
    if name not in _spam_cache:
        s = Spam(name)
        _spam_cache[name] = s
    else:
        s = _spam_cache[name]
    return s


a = get_spam('foo')
b = get_spam('bar')
a is b
# False
c = get_spam('foo')
a is c
# True



# 编写一个工厂函数来修改普通的实例创建行为通常是一个比较简单的方法，但我们能否找到更优雅的解决方案呢？
# 可以考虑重新定义类的__new__方法
# Note: this code doesn't quite work
import weakref
class Spam:
    # 弱引用可以保证实例在没有引用后被回收
    _spam_cache = weakref.WeakKeyDictionary()
    def __new__(cls, name):
        if name in cls._spam_cache:
            return cls._spam_cache[name]
        else:
            self = super().__new__(cls)
            cls._spam_cache[name] = self
            return self
    def __init__(self, name):
        print('Initializing Spam')
        self.name = name
  
# 这样可以达到效果，但是__init__每次都会被调用
        
# 上述代码用到了一个全局变量，并且工厂函数跟类放在一块，我们可以通过将缓存代码放到一个单独的缓存管理器中
import weakref

class CachedSpamManager:
    def __init__(self):
        self._cache = weakref.WeakKeyDictionary()
    
    def get_spam(self, name):
        if name not in self._cache:
            s = Spam(name)
            self._cache[name] = s
        else:
            s = self._cache[name]
        return s
    
    def clear(self):
        self._cache.clear()
    
class Spam:
    # 这样我们如果需要增加更多的的缓存机制，只需要替代manger即可
    manager = CachedSpamManager()
    # 为了防止用户调用这个类实例化，有两种方法
    # 第一个是将类的名字修改为下划线开头，提示用户别直接调用它
    # 第二个是让这个类的init方法抛出一个异常，让它不能被初始化
    def __init__(self, name):
        # self.name = name
        raise RuntimeError("can't instantiate directly")
    # alternate constructor
    # 强调用户别直接调用
    def __new__(cls, name):
        self = cls.__new__(cls)
        self.name = name
    

    

def get_spam(name):
    return Spam.manager.get_spam(name)

# 修改缓存器代码，使用spam._new来创建实例，
class CachedSpamManager:
    def __init__(self):
        self._cache = weakref.WeakKeyDictionary()

    def get_spam(self, name):
        if name not in self._cache:
            s = Spam._new(name)
            self._cache[name] = s
        else:
            s = self._cache[name]
        return s
    
    def clear(self):
        self._cache.clear()