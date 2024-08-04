# 一个根据状态的不同来执行不同操作的的连接对象
class Connection:
    '''
    普通方案，好多个判断语句，效率低下～
    '''
    def __init__(self):
        self.state = 'CLOSED'
    
    def read(self):
        if self.state != "OPEN":
            raise RuntimeError('Not open')
        print('reading')

    def write(self):
        if self.state != "OPEN":
            raise RuntimeError('Not open')
        print('writing')
    
    def open(self):
        if self.state == "OPEN":
            raise RuntimeError('already open')
        self.state = 'OPEN'
    
    def close(self):
        if self.state == "CLOSED":
            raise RuntimeError('already closed')
        self.state = 'CLOSED'
# 传统的多状态的操作对象：代码很复杂，很多的条件判断，执行效率较低，
# 一些常见的read，write操作每次执行前都要执行检查
# 为每个状态定义一个对象
class Connection1:
    '''新方案--对每个状态定义一个类'''
    def __init__(self):
        self.new_state(ClosedConnectionState)

    def new_state(self, newState):
        self._state = newState
        # Delegate to the state class
    
    def read(self):
        return self._state.read(self)
    
    def write(self, data):
        return self._state.write(self, data)
    
    def open(self):
        return self._state.open(self)
    
    def close(self):
        return self._state.close(self)
    
# connection state base class
# 定义一堆待实现的抽象类，看起来是为了约束子类会实现对应方法，否则抛错？
# ConnectionState 可以写成一具体的基类，也可以写成一个抽象基类
class ConnectionState:
    @staticmethod
    def read(conn):
        raise NotImplementedError()
    
    @staticmethod
    def write(conn, data):
        raise NotImplementedError()
    
    @staticmethod
    def open(conn):
        raise NotImplementedError()
    
    @staticmethod
    def close(conn):
        raise NotImplementedError()
    
# implementation of different states
class ClosedConnectionState(ConnectionState):
    @staticmethod
    def read(conn):
        raise RuntimeError('Not open')
    
    @staticmethod
    def write(conn, data):
        raise RuntimeError('Not open')
    
    @staticmethod
    def open(conn):
        conn.new_state(OpenConnectionState)

    @staticmethod
    def close(conn):
        raise RuntimeError('Already closed')
    
class OpenConnectionState(ConnectionState):
    @staticmethod
    def read(conn):
        print('reading')
    
    @staticmethod
    def write(conn, data):
        print('writing')
    
    @staticmethod
    def open(conn):
        raise RuntimeError('Already open')
    
    @staticmethod
    def close(conn):
        conn.new_state(ClosedConnectionState)

# 使用演示
c = Connection1()
c._state
# <class '__main__.ClosedConnectionState'>
c.read()
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "example.py", line 10, in read
#         return self._state.read(self)
#     File "example.py", line 43, in read
#         raise RuntimeError('Not open')
# RuntimeError: Not open
c.open()
c._state
# <class '__main__.OpenConnectionState'>
c.read()
# reading
c.write('hello') 
# writing
c.close() 
c._state
# <class '__main__.ClosedConnectionState'>

# 如果代码中出现太多的条件判断语句的话，代码就会变得难以维护和阅读，
# 这里的解决方案是将每个状态抽取出来定义一个类，且定义该类下的对应方法的行为
# 基类中定义的NotImplementedError是为了确保子类实现了相应的方法
# 设计模式中有一种模式叫状态模式
# 尝试用抽象基类实现ConnectionState
from abc import ABCMeta, abstractmethod
class ConnectionState1(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def read(conn):
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def write(conn, data):
        pass
        # raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def open(conn):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def close(conn):
        raise NotImplementedError()

class ClosedConnectionState1(ConnectionState1):
    @staticmethod
    def read(conn):
        raise RuntimeError('Not open')
    
    @staticmethod
    def write(conn, data):
        raise RuntimeError('Not open')
    
    @staticmethod
    def open(conn):
        conn.new_state(OpenConnectionState1)

    @staticmethod
    def close(conn):
        raise RuntimeError('Already closed')
    
class OpenConnectionState1(ConnectionState1):
    @staticmethod
    def read(conn):
        print('reading')
    
    @staticmethod
    def write(conn, data):
        print('writing')
    
    @staticmethod
    def open(conn):
        raise RuntimeError('Already open')
    
    @staticmethod
    def close(conn):
        conn.new_state(ClosedConnectionState1)

class Connection2:
    '''新方案--对每个状态定义一个类'''
    def __init__(self):
        self.new_state(ClosedConnectionState1)

    def new_state(self, newState):
        self._state = newState
        # Delegate to the state class
    
    def read(self):
        return self._state.read(self)
    
    def write(self, data):
        return self._state.write(self, data)
    
    def open(self):
        return self._state.open(self)
    
    def close(self):
        return self._state.close(self)
    
# 使用演示
c = Connection2()
c._state
# <class '__main__.ClosedConnectionState'>
c.read()
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "example.py", line 10, in read
#         return self._state.read(self)
#     File "example.py", line 43, in read
#         raise RuntimeError('Not open')
# RuntimeError: Not open
c.open()
c._state
# <class '__main__.OpenConnectionState'>
c.read()
# reading
c.write('hello') 
# writing
c.close() 
c._state


from abc import ABC, abstractmethod

class MyBaseClass(ABC):

    @abstractmethod
    def someMethod(self):
        pass

class MySubClass(MyBaseClass):
    pass

instance = MySubClass()  # This will raise an error

# 你有一个字符串形式的方法名称，想通过它调用某个对象对应方法
#  getattr()
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return 'Point({!r:},{!r:})'.format(self.x, self.y)
    
    def distance(self, x, y):
        return math.hypot(self.x - x, self.y - y)
p = Point(2,3)
d = getattr(p, 'distance')(0, 0) # calls p.distance(0,0)
# 另一种方法是使用operator.methodcaller()
import operator
operator.methodcaller('distance', 0, 0)(p)
# 当你需要通过相同的参数多次调用某个方法时，使用operator.methodcaller就很方便了，比如你需要排序一系列的点
points = [
    Point(1,2),
    Point(3,0),
    Point(10,-3),
    Point(-5,-7),
    Point(-1,8),
    Point(3,2),
]
# sort by distance from origin(0,0)
points.sort(key=operator.methodcaller('distance', 0, 0))
# 创建一个可调用对象，然后调用的时候传递相关参数，就可以实现多次调用
p = Point(3,4)
d = operator.methodcaller('distance', 0,0)
d(p) # 5.0
# 通过方法名称字符串来调用方法通常出现在需要模拟case语句或者实现访问者模式的时候
# 假设你要写一个数学表达式的程序
class Node:
    pass

class UnaryOperator(Node):
    def __init__(self, operand):
        self.operand = operand
    
class BinaryOperator(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Add(BinaryOperator):
    pass

class Sub(BinaryOperator):
    pass

class Mul(BinaryOperator):
    pass

class Div(BinaryOperator):
    pass

class Negate(UnaryOperator):
    pass

class Number(Node):
    def __init__(self, value):
        self.value = value

# 利用这些类构建嵌套数据结构
t1 = Sub(Number(3), Number(4))
t2 = Mul(Number(2), t1)
t3 = Div(t2, Number(5))
t4 = Add(Number(1), t3)

# 对于每个表达式，每次都要重新定义一遍
# 我们可以借助访问者模式，设计一种更通用的方式让它支持所有的数字和操作符
# 这个访问者模式的作用是将散落在每个函数里的功能逻辑，抽象到一个地方去定义和管理？
class NodeVisitor:
    def visit(self, node):
        methname = 'visit_' + type(node).__name__
        meth = getattr(self, methname, None)
        if meth is None:
            meth = self.generic_visit
        return meth(node)
    def generic_visit(self, node):
        raise RuntimeError('No {} method'.format('visit_' + type(node).__name__))
# 定义一个类继承它并且实现各种visit_Name方法,Name是node类型
class Evaluator(NodeVisitor):
    def visit_Number(self, node):
        return node.value
    
    def visit_Add(self, node):
        return self.visit(node.left) + self.visit(node.right)
    
    def visit_Sub(self, node):
        return self.visit(node.left) - self.visit(node.right)
    
    def visit_Mul(self, node):
        return self.visit(node.left) * self.visit(node.right)
    
    def visit_Div(self, node):
        return self.visit(node.left) / self.visit(node.right)
    
    def visit_Negate(self, node):
        return -node.operand
    
e = Evaluator()
e.visit(t4) # 0.6
    
# 定义一个类在一个栈上面讲一个表达式转换成多个操作序列
class StackCode(NodeVisitor):
    def generate_code(self, node):
        self.instructions = []
        self.visit(node)
        return self.instructions
    
    def visit_Number(self, node):
        self.instructions.append(('PUSH', node.value))
    
    # 处理两个操作对象的运算
    def binop(self, node, instruction):
        self.visit(node.left)
        self.visit(node.right)
        self.instructions.append((instruction,))

    def visit_Add(self, node):
        self.binop(node, 'ADD')
    
    def visit_Sub(self, node):
        self.binop(node, 'SUB')
    
    def visit_Mul(self, node):
        self.binop(node, 'MUL')
    
    def visit_Div(self, node):
        self.binop(node, 'DIV')
    
    def unaryop(self, node. instruction):
        self.visit(node.operand)
        self.instructions.append((instruction,))

    def visit_Negate(self, node):
        self.unaryop(node, 'NEG')

# 使用访问者模式的好处就是通过getattr来获取相应的方法，并利用递归来遍历所有的节点
def binop(self, node, instruction):
    self.visit(node.left)
    self.visit(node.right)
    self.instructions.append((instruction,))

# 这种技术也是实现其他语言中的switch或case语句的方式，
# 如果你正在写一个http框架，你可能会写这样一个请求分发的控制器
class HTTPHandler:
    def handle(self, request):
        methname = 'do_' + request.request_method
        getattr(self, methname)(request)
    def do_GET(self, request):
        pass
    def do_POST(self, request):
        pass
    def do_HEAD(self, request):
        pass
        

# 访问者模式一个缺点就是它严重依赖地柜，如果数据结构嵌套层次太深可能会有问题，有时会超过python的递归深度限制(sys.getrecursionlimit(): 1000)
# 可以使用生成器或迭代器来实现非递归遍历算法
# 在跟解析和编译相关的编程中使用访问者模式是很常见的，python的ast模块中的去看看源码
    




