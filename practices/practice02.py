# 迭代器与生成器
# 迭代不只是处理序列中元素的一种方法，还可以创建自己的迭代对象，再itertools模块中使用有用的迭代模式，构造生成器函数等
# 手动读取一个文件的所有行
def manual_iter():
    with open('/etc/passwd') as f:
        try:
            while True:
                line = next(f)
                # 可以通过返回一个指定值来标记结尾
                if line is None:
                    break
                print(line, end='')
        # 通常指示迭代的结尾
        except StopIteration:
            pass

items = [1,2,3]
it = iter(items)
next(it) # 1
next(it) # 2
next(it) # 3

# 构建一个自定义容器对象，里面包含列表，元组或其他可迭代对象，
# 如果想直接在这个容器对象上执行迭代操作需要定义一个__inter方法，将迭代操作代理到容器内部的对象上去
class Node:
    def __init__(self, value):
        self._value = value
        self._children = []
    def __repr__(self):
        return 'Node({!r})'.format(self._value)
    
    def add_child(self, node):
        self._children.append(node)

    # 将代理操作对象映射到内部的对象上去
    def __iter__(self):
        return iter(self._children)
# example
if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    # outputs Node(1), Node(2)
    for ch in root:
        print(ch)
# 自定义一个支持自定义步进的累加器
def frange(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment

# 可以使用for循环或者其他接受一个可迭代对象的函数（比如sum）
for n in frange(0, 4, 0.5):
    print(n)

# 一个函数需要有一个yield语句即可将其转换为一个生成器
# 跟函数不同的是，生成器只能用于迭代操作
def countdown(n):
    print('staring to count from', n)
    while n > 0:
        yield n
        n -= 1
    print('Done!')

c = countdown(3)
next(c)
# 3
next(c)
# 2
next(c)
# 1
next(c)
# Done!
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
# StopIteration

# 一个生成器函数的主要特征是它只会在迭代操作中使用到next操作，一旦生成器函数退出
# 迭代终止，我们在迭代中通常使用for语句自动处理这些细节无需度担心  

# 实现一个以深度优先方式遍历树形节点的生成器
class Node:
    def __init__(self, value) -> None:
        self._value = value
        self._children = []

    def __repr__(self) -> str:
        return 'Node({!r})'.format(self._value)
    
    def add_child(self, node):
        self._children.append(node)
    # 自定义迭代器
    def __iter__(self):
        return iter(self._children)
    
    # 通过yield将你的迭代器定义成一个生成器，使其能完成迭代功能
    def depth_first(self):
        yield self
        for c in self:
            yield from c.depth_first()

if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(Node(3))
    child1.add_child(Node(4))
    child2.add_child(Node(5))

    for ch in root.depth_first():
        print(ch)

# python协议支持__iter__() 方法返回一个特殊的迭代器对象，这个迭代器对象实现了__next__()方法并通过StopIteration异常标识迭代的完成
# 但是，实现这些通常会比较繁琐，可以使用一个关联迭代器重新实现depth_first()方法
class Node2:
    def __init__(self, value) -> None:
        self._value = value
        self._children = []
    
    def __repr__(self) -> str:
        return 'Node({!r})'.format(self._value)
    
    def add_child(self, node):
        self._children.append(node)
    # 自定义迭代器
    def __iter__(self):
        return iter(self._children)
    
    def depth_first(self):
        return DepthFirstIterator(self)
    
class DepthFirstIterator(object):
    '''
    depth-first traversal
    '''
    def __init__(self, start_node) -> None:
        self._node = start_node
        self._children_iter = None
        self._child_iter = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # return myself if just started,create an iterator for children
        if self._children_iter is None:
            self._children_iter = iter(self._node)
            return self._node
        # if processing a child, return its next item
        elif self._child_iter:
            try:
                nextchild = next(self._child_iter)
                return nextchild
            except StopIteration:
                self._child_iter = None
                return next(self)
        # advance to the next child and start its iteration
        else:
            # 通过取出下一个待迭代元素并调用自身完成迭代
            self._child_iter = next(self._children_iter).depth_first()
            return next(self)

# 反向迭代reversed，仅当对象的大小可预先确定或者对象实现了__reversed__()的特殊方法才能生效
# 否则，你必须先讲对象转换为一个列表才行,若可迭代元素很多的话，将其预先转换为一个列表需要消耗大量的内存
# print a file backwards
f = open('somefile')
for line in reversed(list(f)):
    print(line, end='')

# 自定义一个反向迭代器可以使代码非常的高效，因为它不再需要讲数据填充到一个列表中然后反向迭代这个列表
class Countdown:
    def __init__(self, start) -> None:
        self.start = start

    # Forward iterator
    def __iter__(self):
        n = 1
        while n <= self.start:
            yield n
            n += 1   
    # Reverse iterator
    def __reversed__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1     

for rr in reversed(Countdown(30)):
    print(rr)

for rr in Countdown(30):
    print(rr)

# 带有外部状态的生成器函数
from collections import deque

class linehistory:
    def __init__(self, lines, histlen=3) -> None:
        self.lines = lines
        self.history = deque(maxlen=histlen)
    
    def __iter__(self):
        for lineno,line in enumerate(self.lines, 1):
            self.history.append(lineno, line)
            yield line
    def clear(self):
        self.history.clear()

# 创建一个实例对象，访问内部属性值
with open('somefile.txt') as f:
    lines = linehistory(f)
    for line in lines:
        if 'python' in line:
            for lineno, hline in lines.history:
                print('{}:{}'.format(lineno, hline), end='')
# 如果你在迭代操作不使用for循环语句，那么你的先调用iter函数
                
# 迭代器和生成器不能使用标准的切片操作，因为它们的长度事先不知道且没有实现索引
# 函数islice返回一个可以生成指定元素的迭代器，需要注意的是islice会消耗掉传入的迭代器中的数据，这个操作是不可逆的
# 如果想要之后再次访问这个迭代器，则需要先将它的数据放到列表中
def count(n):
    while True:
        yield n
        n += 1
c = count(0)

import itertools

for x in itertools.islice(c, 10, 20):
    print(x)
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17
# 18
# 19
    
# itertools.dropwhile()会返回一个迭代器对象，丢弃原有序列中直到函数返回false之前的所有元素，然后返回后面所有元素
# 如跳过开始部分的注释行
# itertools提供的相关方法适用于所有可迭代对象，包括哪些实现不能确定大小的，比如生成器、文件及其类似的对象
from itertools import dropwhile
with open('/etc/passwd') as f:
    for line in dropwhile(lambda line: not line.startswith('#'), f):
        print(line, end='')

# 如果明确知道要跳过元素的序号，可以使用itertools.islice()来代替
from itertools import islice
items = ['a', 'b', 'c', 1, 4, 10, 15]
for line in islice(items, 3, None):
    print(line, end='')

# 注意dropwhile方法跟filter是不一样的，区别类似于正则表达式里的/g
with open('/etc/passwd') as f:
    lines = (line for line in f if not line.startswith('#'))
    for line in lines:
        print(line, end='')

# itertools.permutations() ：获取一个集合的所有可能的排列或组合       
# itertools.combinations() ：获取一个集合的所有可能的排列或组合，忽略元素顺序，ab 和 ba 只会输出一个
# itertools.combinations_with_replacement() ：获取一个集合的所有可能的排列或组合，允许同一个元素被选择多次
# 如果遇到比较复杂的迭代问题，可以去看看itertools模块里有没有提供相关的工具方法
        
# enumerate()函数可以在迭代一个序列的同时跟踪正在被处理的元素索引
my_list = ['a', 'b', 'c']
for idx, val in enumerate(my_list, 2):
    print(idx, val)

# 遍历文件中想在错误消息中使用行号定位
def parse_data(filename):
    with open(filename, 'rt') as f:
        for lineno, line in enumerate(f, 1):
            fields = line.split()
            try:
                count = int(fields[1])
                # ...
            except ValueError as e:
                print('Line {}: Parse error: {}'.format(lineno. e))
# enumerate 也可以很容易的辅助计数
# 在一个已经解压后的元组序列上使用enumerate函数很容易掉入陷阱
data = [ (1, 2), (3, 4), (5, 6), (7, 8) ]
for n,(x,y) in enumerate(data):
    # ...
    n # 可以作为计数变量

# 如果想要迭代多个序列，可以使用zip方法,zip的迭代长度会与短序列长度报纸一致
xpts = [1, 5, 4, 2, 10, 7]
ypts = [101, 78, 37, 15, 62, 99]

for x,y in zip(xpts, ypts):
    print(x, y)
# 如果希望与长序列的长度保持一致，可以使用zip_longest来替代
# zip()很适合成对处理数据
headers = ['name', 'shares', 'price']
values = ['ACME', 100, 490.1]
s = dict(zip(headers, values)) 

for name, val in zip(headers, values):
    print(name, '=', val)
# zip 会创建一个爹大气来作为结果返回，如果需要讲结对值存储在列表中，可以使用list
list(zip(headers, values))

# chain() 用于对不同的集合中的所有元素执行某些操作
active_items = set()
inactive_items = set()
from itertools import chain
for item in chain(active_items, inactive_items):
    # Process item
    print(item)
# 合并操作会要求不同集合的数据类型益智，且会创建一个全新的序列
# chain 比单纯的合并操作高效的多，不需要创建全新的序列、省内存。可容纳不同类型的集合
for x in active_items + inactive_items:
    #...
    print(item)
for x in chain(active_items, inactive_items):
    #...
    print(item)

# yield语句作为数据的生产者而for循环语句作为数据的消费者
# 这些生成器被连在一起后，每个yield会产生一个单独的数据元素传递给迭代管道的下一阶段
    
#
import os
import fnmatch
import gzip
import bz2
import re

def gen_find(filepat, top):
    '''
    find all filenames in a directory tree that match a shell wildcard pattern
    '''
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            yield os.path.join(path, name)

def gen_opener(filenames):
    '''
    open a sequence of filenames one at a time producing a file object.
    the file is closed immediately when proceeding to the next iteration
    '''
    for filename in filenames:
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rt')
        elif filename.endswith('.bz2'):
            f = bz2.open(filename, 'rt')
        else:
            f = bz2.open(filename, 'rt')
        # 文件的读取结果
        yield f
        f.close()

def gen_concatenate(iterators):
    '''
    chain a sequence of iterators together into a single sequence.
    '''
    for it in iterators:
        yield from it

def gen_grep(pattern, lines):
    '''
    look for a regex pattern in a sequence of lines
    '''
    pat = re.compile(pattern)
    for line in lines:
        if pat.search(line):
            yield line

# 基于基础的工具函数，创建一个处理管道，查找包含单词python的所有日志行
lognames = gen_find('access-log*', 'www')
files = gen_opener(lognames)
lines = gen_concatenate(files)
pylines = gen_grep('(?i)python', lines)

# 扩展管道可以在生成器表达式中包装数据
lognames = gen_find('access-log*', 'ww')
files = gen_opener(lognames)
lines = gen_concatenate(files)
pylines = gen_grep('(?i)python', lines)
bytecolumn = (line.rsplit(None, 1)[1] for line in pylines)
bytes = (int(x) for x in bytecolumn if x != '-')
# sum函数是最终的程序启动者
print('total', sum(bytes))


# 将一个多层嵌套的列表展开成一个单层列表
from collections import Iterable
def flattern(items, ignore_types=(str, bytes)):
    for x in items:
        # ignore_types 用来排出字符串和字节
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flattern(x)
            # yield from 在生成器中调用其他生成器作为子例程的时候非常有用，如果不使用它的话，就必须写额外的for循环了
            # for i in flattern(x):
            #     yield i
        else:
            yield x
items = [1, 2, [3, 4, [5, 6], 7], 8]
# Produces 1 2 3 4 5 6 7 8
for x in flatten(items):
    print(x)

# yield from 在涉及到基于协程和生成器中并发编程中扮演着更加重要的角色

# heapq.merge可以用来排序多个有序序列，它不会立马读所有序列，也不会对输入做任何的排序检测
# 仅仅是检查所有序列中的开始部分并返回最小的那个。它只是引用传入的序列
import heapq
with open('sorted_file_1', 'rt') as file1, \
    open('sorted_file_2', 'rt') as file2, \
    open('merged_file', 'wt') as outf:

    for line in heapq.merge(file1, file2):
        outf.write(line)

# 可以通过约定iter的callable来代替while循环来迭代处理数据
# 如遇到文件结束符则终止读取文件
CHUNKSIZE = 8192
def reader(s):
    while True:
        data = s.recv(CHUNKSIZE)
        if data == b'':
            break
        # process_data(data)

# 使用iter代替while
def reader2(s):
    for chunk in iter(lambda: s.recv(CHUNKSIZE), b''):
        pass
        # process_data(data)