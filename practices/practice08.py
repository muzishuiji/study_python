# 巧妙的使用生成器在树遍历或搜索算法中消除递归
import types
class Node:
    pass
class NodeVisitor:
    # 用来做调度
    def visit(self, node):
        stack = [node]
        last_result = None
        while stack:
            try:
                last = stack[-1]
                # 如果是一个生成器就将结果入栈
                if isinstance(last, types.GeneratorType):
                    stack.append(last.send(last_result))
                    last_result = None
                elif isinstance(last, Node):
                    stack.append(self._visit(stack.pop()))
                else:
                    last_result = stack.pop()

            except StopIteration:
                stack.pop()
        return last_result
    # 用来做访问
    def _visit(self, node):
        methname = 'visit_' + type(node).__name__
        meth = getattr(self, methname, None)
        if meth is None:
            meth = self.generic_visit
        return meth(node)
    # 用来做异常捕获
    def generic_visit(self, node):
        raise RuntimeError('No {} method'.format('visit_' + type(node).__name__))
  
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


# A sample visitor class that evaluates expressions
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
        return -self.visit(node.operand)
    
if __name__ == '__main__':
    # 1 + 2*(3-4) / 5
    t1 = Sub(Number(3), Number(4))
    t2 = Mul(Number(2), t1)
    t3 = Div(t2, Number(5))
    t4 = Add(Number(1), t3)
    # evaluate it
    e = Evaluator()
    print(e.visit(4)) # Outputs 0.6

# 如果起那套层次太深，上述的evaluator会失效
a = Number(0)
for n in range(1, 100000):
    a = Add(a, Number(n))

e = Evaluator()
e.visit(a)
# Traceback (most recent call last):
# ...
#     File "visitor.py", line 29, in _visit
# return meth(node)
#     File "visitor.py", line 67, in visit_Add
# return self.visit(node.left) + self.visit(node.right)
# RuntimeError: maximum recursion depth exceeded

# 改写为通过生成器实现evaluator
class Evaluator(NodeVisitor):
    def visit_Number(self, node):
        return node.value
    
    def visit_Add(self, node):
        yield (yield node.left) + (yield node.right)

    def visit_Sub(self, node):
        yield (yield node.left) - (yield node.right)
    
    def visit_Mul(self, node):
        yield (yield node.left) * (yield node.right)

    def visit_Div(self, node):
        yield (yield node.left) / (yield node.right)

    def visit_Negate(self, node):
        yield - (yield node.operand)

# 在此运行，就不会报错了
a = Number(0)
for n in range(1, 100000):
    a = Add(a, Number(n))

e = Evaluator()
e.visit(a) 
# 4999950000

# 生成器和协程在程序控制流方面功能强大
# 避免递归的一个常用方法是使用一个栈或者对类的数据结构，例如，深度优先的遍历算法，第一次碰到一个节点将其压入栈中，处理完后弹出栈
# visit方法的核心思路就是这样
# 当碰到yield语句时，生成器会返回一个数据并暂时挂起
# 没有太明白为什么生成器技术可以代替递归

# 简单的循环引用，双亲节点有指针指向孩子节点，孩子节点又返回来指向双亲节点，这种情况下，可以考虑weakref库中的弱引用
import weakref
class Node:
    def __init__(self, value):
        self.value = value
        self._parent = None
        self.children = []
    
    def __repr__(self):
        return 'Node({!r:})'.format(self.value)

    # property that manages the parent as weak-reference
    @property
    def parent(self):
        return None if self._parent is None else self._parent()
    
    @parent.setter
    def parent(self, node):
        self._parent = weakref.ref(node)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

root = Node('parent')
c1 = Node('child')
root.add_child(c1)
print(c1.parent)
# Node('parent')
del root
print(c1.parent)
# None

# 循环引用的数据结构在python中是一个很棘手的问题，因为正常的垃圾回收机制不能适用于这种情形
# class just to illustrate when deletion occurs
class Data:
    def __del__(self):
        print('Data.__del__')
# Node class involving a cycle
class Node:
    def __init__(self):
        self.data = Data()
        self.parent = None
        self.children = []
    def add_child(self, child):
        self.children.append(child)
        child.parent = self


a = Data()
del a # immediately deleted
# Data.__del__
del a # immediately deleted
# Data.__del__
a = Node()
a.add_child(Node())
del a # not deleted(no message)
# 最后一个删除的打印语句没有出现，python的垃圾回收机制是基于简单的引用计数
# 当一个对象的引用数变成0的时候才会立即删除掉，对于循环引用这个条件永远不会成立
# python还有另外的垃圾回收器来专门针对循
# 环引用的，但是你永远不知道它什么时候会触发，你可以手动触发它
import gc
gc.collect() # force collection
# 如果循环引用的对象自己定义了_del_ 方法，会导致垃圾回收永远不会回收这个对象，导致内存泄漏,强制内存回收也会失败
# Node class involving a cycle
class Node:
    def __init__(self):
        self.data = Data()
        self.parent = None
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    # never define like this
    def __del__(self):
        del self.data
        del self.parent
        del self.children

# 可以用弱引用来解决循环引用的问题
import weakref
a = Node()
a.add_child(Node())
a_ref = weakref.ref(a)
a_ref
# <weakref at 0x100581f70; to 'Node' at 0x1005c5410>
# 当删除原对象时，弱引用会被回收，因为原始对象的引用计数没有增加
print(a_ref())
# <__main__.Node object at 0x1005c5410>
del a
# Data.__del__
print(a_ref())
# None