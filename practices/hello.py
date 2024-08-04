data = [ 'ACME', 50, 91.1, (2012, 12, 21) ]
_, shares, price, _ = data

# 计算公司前8个月销售数据的序列
# *trailing_qtrs, current_qtr = sales_record
# trailing_avg = sum(trailing_qtrs) / len(trailing_qtrs)
# return avg_comparison(trailing_avg, current_qtr)

# 值得注意的是，星号表达式在迭代元素为可变长元组的序列时是很有用的
records = [
    ('foo', 1, 2),
    ('bar', 'hello'),
    ('foo', 3, 4),
]

def do_foo(x, y):
    print('foo', x, y)

def do_bar(s):
    print('bar', s)

for tag, *args in records:
    if tag == 'foo':
        do_foo(*args)
    elif tag == 'bar':
        do_bar(args)
# 星号解压法在字符串操作的时候会很有用，比如字符串的分隔

line = 'nobody:*:-2:-2:Unprivileged User:/var/empty:/usr/bin/false'
uname, *fields, homedir, sh = line.split(':')

# 如果你希望解压一些元素后丢弃它们，可以使用一个普通的废弃名称，如_或者ign(ignore)

record = ('ACME', 50, 123.45, (12, 18, 2012))
name, *_, (*_, year) = record
year # 2012

# 你甚至可以借助这种分割语法实现递归运算
items = [1, 10, 7, 4, 5, 9]
def sum_num(items):
    if not items:  # 处理递归结束的情况
        return 0
    head, *tail = items
    return head + sum_num(tail)

result = sum_num(items)
print(result)


from collections import deque

# 搜索过程代码～
def search (lines, pattern, history = 5):
    # deque(maxlen = N)构造函数回创建一个固定大小的队列，当新的元素加入并且这个队列已满的时候，最老的元素会自动被移除掉
    # 这样说的话，这个数据结构很适合做窗口控制
    # deque类可以被用在任何你只需要一个简单队列数据结构的场合，如果不设置最大队列大小，就会得到一个无线大小队列，可以在队列的两端执行添加和弹出元素的操作
    # 不同于列表的是，做个结构在两端插入或删除元素的时间复杂度都是O(1),普通列表的开头插入或删除元素的时间复杂度是O(N)
    previous_line = deque(maxlen=history)
    for line in lines:
        if pattern in line:
            yield line, previous_line
        previous_line.append(line)

# 搜索结果代码
# example use on a file
if __name == '__main__':
    with open(r'./file.txt') as f:
        for line, prevlines in search(f, 'python', 5):
            for pline in prevlines:
                print(pline, end='')
            print(line, end='')
            print('-' * 20)

# 查找最大或最小的N个元素列表
import heapq
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nlargest(3, nums))  # Prints [42, 37, 23]
print(heapq.nsmallest(3, nums)) # Prints [-4, 1, 2]

# 还可以实现自定义的排序函数
portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
# 最终结果会以price的值排序
cheap = heapq.nsmallest(3, portfolio, key=lambda s:s['price'])
expensive = heapq.nsmallest(3, portfolio, key=lambda s:s['price'])
# 底层实现的查找函数提供了较好的性能，首先会将集合数据进行堆排序放到一个列表中
# 堆数据结构最重要的特征是heap[0]永远是最小的元素，便给剩余元素可以很容易的通过高调用heapq.heappop()得到
# heapq.heappop(heap) 最小的第一个元素
# heapq.heappop(heap) 最小的第二个元素
# heapq.heappop(heap) 最小的第三个元素
# 当获取的N和集合大小接近的时候，使用切片操作会更快点。查找最大或最小使用min或max更快些
# 使用heapq模块实现一个简单的优先级队列
import heapq
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1
    def pop(self):
        # 取值的时候会根据优先级来取
        return heapq.heappop(self._queue)[-1]
    
# pop操作会返回优先级最高的元素，优先级相同的会按照被插入到队列中的顺序返回
# priority 设置为负数是为了保证元素按照优先级从高到底排序
# index是为了保证同等优先级的元素的正确排序，index变量也在相同优先级元素比较的时候起到重要作用
# 如果你需要在多个线程使用同一个队列，你需要增加适当的锁和信号量机制
    
# 你可以很方便的使用collections模块中的defaultdict来构造这样的字典
# defaultdict的一个特征是它会自动初始化每个key刚开始对应的值，你只需要关注添加元素操作
from collections import defaultdict
d = defaultdict(list)
d['a'].append(1)
d['a'].append(2)
d['b'].append(4)
print(d)

d = defaultdict(set)
d['a'].append(1)
d['a'].append(2)
d['b'].append(4)
print(d)

# 需要注意的是，defaultdict会自动为将要访问的键创建映射实体，如果不需要，可以在一个普通的字典上使用setdefault方法来替代
d = {}
d.setdefault('a', []).append(1)
d.setdefault('a', []).append(2)
d.setdefault('b', []).append(4)

# 手动实现 vs 用库函数实现

d = {}
for key, value in pairs:
    if key not in d:
        d[key] = []
    d[key].append(value)

# defaultdict
d = defaultdict(list)
for key, value in pairs:
    d[key].append(value)   

# 通常来说，字典中的顺序是不稳定的，如果你想要保持元素的插入顺序则应该是用列表，而字典可以自动合并重复元素
# 如果你想创建一个字典，并且在迭代或序列化这个字典的时候能够控制元素的顺序,则使用orderdict类
from collections import OrderedDict
d = OrderedDict()
d['foo'] = 1
d['bar'] = 1
d['spam'] = 1
d['grok'] = 1
# Outputs "foo 1", "bar 2", "spam 3", "grok 4"
for key in d:
    print(key, d[key])

# 当你想要构建一个将来需要序列话或编码成其他格式的映射的时候，ordereddict是非常有用的，
# 比如，你想要精准控制json编码后字段的顺序，你可以先试用ordereddict来构建这样的数据
import json
json.dumps(d)
# ordereddict 内部维护着一个根据键插入顺序排序的双向链表，每当一个新的元素进来的时候，它会被放到链表的尾部
# 对一个已经存在的键重复赋值不会改变键的顺序
# 一个ordereddict的大小是一个普通字典的两倍，因为它内部维护着另外一个链表，如果你要构建一个大量ordereddict实例的数据结构
# 的时候，如读取100000行csv数据当到一个ordereddict列表中去，那么你就的仔细权衡一下是否使用ordereddict带来的好处要大过额外内存消耗的影响
# 对字典进行运算，需要借助zip将键和值反过来
prices = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}
min_price = min(zip(prices.values(), prices.keys()))
# min_price is (10.75, 'FB')
max_price = max(zip(prices.values(), prices.keys()))
# max_price is (612.78, 'AAPL')
# 类似的，可以使用zip()或sorted来排列字典数据
prices_sorted = sorted(zip(prices.values(), prices.keys()))
# prices_sorted is [(10.75, 'FB'), (37.2, 'HPQ'),
#                   (45.23, 'ACME'), (205.55, 'IBM'),
#                   (612.78, 'AAPL')]
# 需要注意的是，zip创建的是一个只能访问一次的迭代器
prices_and_names = zip(prices.values(), prices.keys())
print(min(prices_and_names)) # OK
print(max(prices_and_names)) # ValueError: max() arg is an empty sequence

# 如果你在一个字典上执行普通的数学运算，你会发现它们仅仅作用于键而不是值
min(prices) # # Returns 'AAPL'
max(prices) # Returns 'IBM'  
# 可以通过制定获取对应的键的信息来完成数字运算
min(prices, key=lambda k:prices[k]) # returns "FB"
max(prices, key=lambda k:prices[k]) # returns "AAPL"
# 通常基于数值排序更直接的方法是通过zip反转key和value，然后再去做数学运算

# 如果对应值的信息是相等的，则会根据键排序的结果返回
prices = { 'AAA' : 45.23, 'ZZZ': 45.23 }
min(zip(prices.values(), prices.keys())) # returns (45.23, 'AAA')
max(zip(prices.values(), prices.keys())) # returns (45.23, 'ZZZ')
# 寻找两个字典的相同点，可以简单在两字典的keys或者items方法返回结果上执行集合操作
a = {
    'x' : 1,
    'y' : 2,
    'z' : 3
}

b = {
    'w' : 10,
    'x' : 11,
    'y' : 2
}
# find keys in common
a.keys() & b.keys() # {'x','y'}
# find keys i a that are not in b
a.keys() - b.keys() # {'z'}
# find (key, value) pairs in common
a.items() & b.items() # {('y', 2)}
a.items() | b.items() # {('x', 1), ('w', 10), ('y', 2), ('z', 3), ('x', 11)}
# 集合操作也可以用于修改/过滤字典元素
c = {key:a[key] for key in a.keys() - {'z', 'w'}}
# c is {'x': 1, 'y': 2}
# 不能直接对values()的结果执行集合操作，如果真的有需要，可以考虑先将值转换成set，再进行结合运算。

# 保持元素顺序的同时消除重复的值
def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)
# 上面的房啊只支持不可hash的元素，如果想要消除不可哈希（dict类型）的重复元素的话，需要改变下
def dedupe(items, key=None):
    seen = set()
    for item in items:
        # 将序列元素转换为hashable类型，key函数模仿了sorted，min，max等内置函数的相似功能
        val = item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(val)
# 使用示例
a = [ {'x':1, 'y':2}, {'x':1, 'y':3}, {'x':1, 'y':2}, {'x':2, 'y':4}]
# 去重x，y都相等的
list(dedupe(a, key=lambda d: (d['x'], d['y'])))
# [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 2, 'y': 4}]
# 或者去重x相等的
list(dedupe(a, key=lambda d: d['x']))
# [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 2, 'y':

# 当你希望读取一个文件，消除重复行，可以这样做
with open('./xx.txt', 'r') as f:
    for line in dedupe(f):
        #...

# 命名切片可以用于在某些固定位置提取内容
record = '....................100 .......513.25 ..........'
cost = int(record[20:23]) * float(record[31:37])

# 命名切片,这样可以避免使用大量难以理解的硬编码下标，这使得你的代码更加清晰可读
SHARES = slice(20, 23)
PRICE = slice(31, 37)
cost = int(record[SHARES]) * float(record[PRICE])
# 如果你有一个切片对象a，你可以分别调用它的a.start,a.stop,a.step等属性来获取更多的信息
a = slice(5, 50, 2)
a.start
a.step
a.stop

for i in range(*a.indices(len(s))):
    print(s[i])
# 你可以通过调用切片的indices(size)方法将它映射到一个已知大小的序列上，这个方法返回一个三元组(start, stop, step)
# 所有的值都会被缩小，直到适合这个已知序列的边界为止，这样，使用时就不会出现indexError异常
    
# 获取出现次数最多的元素
words = [
    'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
    'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
    'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
    'my', 'eyes', "you're", 'under'
]
from collections import Counter
words_counts = Counter(words)
# 出现频率最高的3个单词
top_three = words_counts.most_common(3)
print(top_three)

# counter对象可以接受任意的可哈希元素构成的序列对象，在底层实现上，一个counter对象就是一个字典，将元素映射到它出现的次数上。
words_counts['eyes']
words_counts['the']
# 可以手动增加计数
morewords = ['why','are','you','not','looking','in','my','eyes']
for word in morewords:
    words_counts[word] += 1
# 或者直接使用update方法
words_counts.update(morewords)

# Counter实例一个鲜为人知的特性是它们可以很容易的跟数学运算符相结合
a = Counter(words)
# Counter({'eyes': 8, 'the': 5, 'look': 4, 'into': 3, 'my': 3, 'around': 2,
# "you're": 1, "don't": 1, 'under': 1, 'not': 1})
b = Counter(morewords)
# Counter({'eyes': 1, 'looking': 1, 'are': 1, 'in': 1, 'not': 1, 'you': 1,
# 'my': 1, 'why': 1})
c = a + b
# Counter({'eyes': 9, 'the': 5, 'look': 4, 'my': 4, 'into': 3, 'not': 2,
# 'around': 2, "you're": 1, "don't": 1, 'in': 1, 'why': 1,
# 'looking': 1, 'are': 1, 'under': 1, 'you': 1})
d = a - b
# Counter({'eyes': 7, 'the': 5, 'look': 4, 'into': 3, 'my': 2, 'around': 2,
# "you're": 1, "don't": 1, 'under': 1})
# Counter对象在几乎所有需要制表或者计数数据的场合是非常有用的工具，在解决这类问题的时候你应该优先选择它，而不是手动用字典实现
# 字典列表，希望根据某个或者某几个字典字段来排序列表
rows = [
    {'fname': 'Brian', 'lname': 'Jones', 'uid': 1003},
    {'fname': 'David', 'lname': 'Beazley', 'uid': 1002},
    {'fname': 'John', 'lname': 'Cleese', 'uid': 1001},
    {'fname': 'Big', 'lname': 'Jones', 'uid': 1004}
]
from operator import itemgetter
rows_by_fname = sorted(rows, key=itemgetter('fname'))
rows_by_uid = sorted(rows, key=itemgetter('uid'))

print(rows_by_fname)
print(rows_by_uid)

# itemgetter也支持多个key，现根据aa排序，然后再根据dd排序
rows_by_lfname = sorted(rows, key=itemgetter('lname', 'fname'))
print(rows_by_lfname)

# sorted的参数是callable类型，并且从rows中接受一个单一元素，然后返回被用来排序的值，itemgetter函数就是负责创建者callable对象的
# itemgetter有时候也可以用lambda表达式代替,但是itemgetter会稍微快点
rows_by_fname = sorted(rows, key=lambda r: r['fname'])
rows_by_lfname = sorted(rows, key=lambda r: (r['lname'], r['fname']))

min(rows, key = itemgetter('uid'))
max(rows, key = itemgetter('uid'))

# 针对结构体的指定属性的排序
class User:
    def __init__(self, user_id):
        self.user_id = user_id

    def __repr__(self):
        return f"User(id={self.user_id})"

from operator import attrgetter

def sort_notcompare():
    users = [User(23), User(3), User(99)]
    print(sorted(users, key=lambda u: u.user_id))
    # 也可以使用attrgetter来代替lambda函数
    print(sorted(users, key=attrgetter('user_id')))
    by_name = sorted(users, key=attrgetter('last_name', 'first_name'))
    # 适用sort函数的同时也适用min、max


rows = [
    {'address': '5412 N CLARK', 'date': '07/01/2012'},
    {'address': '5148 N CLARK', 'date': '07/04/2012'},
    {'address': '5800 E 58TH', 'date': '07/02/2012'},
    {'address': '2122 N CLARK', 'date': '07/03/2012'},
    {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'},
    {'address': '1060 W ADDISON', 'date': '07/02/2012'},
    {'address': '4801 N BROADWAY', 'date': '07/01/2012'},
    {'address': '1039 W GRANVILLE', 'date': '07/04/2012'},
]
# 先根据date排序，然后根据date分组
from operator import itemgetter
from itertools import groupby
# 先根据date排序，然后根据date分组
rows_by_date = sorted(rows, key=itemgetter('date'))
for date, items in groupby(rows_by_date, key=itemgetter('date')):
    print(date)
    for i in items:
        print(f"   {i}")
# 一个非常重要的步骤是要根据指定的字段将数据排序，因为groupby仅仅检查连续的元素
# 如果事先并没有排序完成的话，分组函数将得不到想要的结果
# 如果你仅仅只是想根据date字段将数据分组到一个大的数据结构中去，并且允许随机访问，那么最好使用defaultdict来构建一个多值字典
from collections import defaultdict
rows_by_date = defaultdict(list)
for row in rows:
    # 有点类似于js的object，根据键值分组
    rows_by_date[row['date']].append(row);        
for r in rows_by_date['07/01/2012']:
 print(r)

# python 3.7及更高的版本创建的字典都是有序的。
# 如果对内存占用不是很关心，直接创建字典会比groupby函数迭代的方式更快一些。
# 输入的数据集不大，可直接过滤
mylist = [1, 4, -5, 10, -7, 2, 3, -1]
[n for n in mylist if n > 0]  
[n for n in mylist if n < 0]  
# 输入的数据集较大，则使用迭代器来过滤
pos = (n for n in mylist if n > 0)
for x in pos:
    print(x)
# 过滤规则较复杂时，可以抽象成函数
values = ['1', '2', '-3', '-', '4', 'N/A', '5']
def is_init(val):
    try:
        x = int(val)
        return True
    except ValueError:
        return False
isVals = list(filter(is_init, values))
# filter 函数创建了一个迭代器，如果你想得到一个列表的话，使用list去转换
print(isVals)
# Outputs ['1', '2', '-3', '4', '5']

mylist = [1, 4, -5, 10, -7, 2, 3, -1]
import math
# 在过滤时转换数据
[math.sqrt(n) for n in mylist if n > 0]
# 将不符合条件的值替换而不是丢弃它们
clip_neg = [n if n > 0 else 0 for n in mylist]

clip_pos = [n if n < 0 else 0 for n in mylist]\

# 当你需要用另一个相关联的序列来过滤某个序列的时候，可以使用irertools.compress()
addresses = [
    '5412 N CLARK',
    '5148 N CLARK',
    '5800 E 58TH',
    '2122 N CLARK',
    '5645 N RAVENSWOOD',
    '1060 W ADDISON',
    '4801 N BROADWAY',
    '1039 W GRANVILLE',
]
counts = [ 0, 3, 10, 4, 1, 7, 6, 1]
# 输出count的值大于5的结果
from itertools import compress
more5 = [n > 5 for n in counts]
# [False, False, True, False, False, True, True, False]
# 过滤返回more5序列中值为true的元素
list(compress(addresses, more5))
# 和filter函数类似，compress函数也是返回一个迭代器，如果需要得到一个列表，需要使用list()来将结果转换为列表类型

# 构造一个字典，是另一个字典的子集
prices = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}
# price > 200
p1 = { key: value for key, value in prices.items() if value > 200}
# key in 指定数组
tech_names = {'AAPL', 'IBM', 'HPQ', 'MSFT'}
p2 = { key: value for key, value in prices.items() if key in tech_names }
# 大多数字典推导能做到的，通过创建一个元组序列然后把它传给dict函数也能实现
p1 = dict((key, value) for key, value in prices.items() if value > 200)
# 字典的推导方式表意更清晰，运行也更快
# 统一功能的实现方式有多种，可以通过计时和性能测试来辅助选择

# 映射名称到序列元素
from collections import namedtuple
Subscriber = namedtuple('Subscriber', ['addr', 'joined'])
sub = Subscriber('jonesy@exapmle', '2012-01-02')
sub.addr
sub.joined
# 尽管namedtuple的实例看起来是一个普通的类实例，但是它跟元组类型是可交换的，支持所有的普通元组操作，比如索引和解压
len(sub)
addr, joined = sub
addr
joined
# 命名元组的一个主要用途是将你的代码从下标操作中解脱出来，如果你的数据库调用中返回了一个很大的元组列表，通过下标去操作其中的元素
# 当你在表中添加了新的列的时候你的代码可能就会出错了，但是如果你使用了命名元组，就不会有这样的顾虑
# 普通元组的代码如下
def compute_cost(records):
    total = 0.0
    for rec in records:
        total += rec[1] * rec[2]
    return total
# 下标操作通常会让代码表意不清晰，并且非常依赖记录的结构，下面是使用命名元组的版本
from collections import namedtuple
Stock = namedtuple('Stock', ['name','shares', 'price'])
def compute_cost(records):
    total = 0.0
    for rec in records:
        s = Stock(*rec)
        total += s.shares * s.price
    return total
# 给Stock添加新的字段
Stock = namedtuple('Stock', ['name','shares', 'price', 'date', 'time'])
s = Stock('ACME', 100, 123.45, '2010-01-01', '2010-01-01 9:30')
s
# 命名元组的实例可以像普通元组一样使用，但是
# 命名元组另一个用途是作为字典的替代，因为字典的存储需要更多的内存空间，如果需要构建一个非常大的包含字典的数据结构，那么使用命名元组会更加高效
# 不想字典那样，一个命名元组是不可更改的
# s.shares = 74
# 如果真的需要改变某个属性的值，可以使用_replace方法，它会创建一个新的命名元组并将对应的字段用心的值替代
s = s._replace(shares=75) # Stock(name='ACME', shares=75, price=123.45)
# 当你的命名元组拥有可选或缺失字段的时候，它是一个非常方便的填充数据的方法
# 
from collections import namedtuple
Stock = namedtuple('Stock', ['name','shares', 'price', 'date', 'time'])
# create a prototype instance
stock_prototype = Stock('', 0, 0.0, None, None)

def dict_to_stock(s):
    return stock_prototype._replace(**s)
a = {'name': 'ACME', 'shares': 100, 'price': 123.45}
dict_to_stock(a) # Stock(name='ACME', shares=100, price=123.45, date=None, time=None)
b = {'name': 'ACME', 'shares': 100, 'price': 123.45, 'date': '12/17/2012'}
dict_to_stock(b) # Stock(name='ACME', shares=100, price=123.45, date='12/17/2012', time=None)
# 如果你的目标是定义一个需要更新很多实例属性的高效数据结构，那么命名元组不是你的最佳选择，这时候你应该考虑定义一个包含__slots__方法的类

# 转换并同时计算
nums = [1,2,3,4,5]
s = sum(x * x for x in nums)

import os
files = os.listdir('dirname')
if any(name.endswith('.py') for name in files):
    print('There be python!')
else:
    print('Sorry, no python.')
# output a tuple as CSV
s = ('ACME', 50, 123.45)
print(','.join(str(x) for x in s))

# 累加数据
portfolio = [
    {'name':'GOOG', 'shares': 50},
    {'name':'YHOO', 'shares': 75},
    {'name':'AOL', 'shares': 20},
    {'name':'SCOX', 'shares': 65}
]
# 生成其表达式： s['shares'] for s in portfolio
# 生成器表达式会以迭代的方式转换数据，更省内存
# Returns: 20
min_shares = min(s['shares'] for s in portfolio)
# Returns {'name': 'AOL', 'shares': 20}
min_shares = min(portfolio, key=lambda s: s['shares'])

# 在多个字典中进行查找，考虑ChainMap类
a = {'x': 1, 'z': 3 }
b = {'y': 2, 'z': 4 }
from collections import ChainMap
c = ChainMap(a, b)
c['x']
c['y']
c['z']
# 当你需要在多个字典中查找数据的时候，ChainMap类是非常有用的，它会把多个字典合并成一个，然后你就
# ChainMap可以合并字典，但这些字典并不是真的合并在一起了，只是内部创建了一个容纳这些字典的列表，并重新定义了一些常见的字典操作来遍历这个列表
# 若出现重复，则第一次出现的映射值会返回
# 对字典的更新或删除操作总是影响列表中的第一个字典（如果删除的属性在第一个字典中不存在，则会报错，即只有查询比较方便和准确了）
values = ChainMap()
values['x'] = 1
# Add a new mapping
values = values.new_child()
values['x'] = 2
# Add a new mapping
values = values.new_child()
values['x'] = 3
values
ChainMap({'x': 3}, {'x': 2}, {'x': 1})
values['x']
#3
# Discard last mapping
values = values.parents
values['x']
#2
# Discard last mapping
values = values.parents
values['x']
# 1
values
# ChainMap({'x': 1})
# 可以借助update来合并字典，但是update会创建新的字典（原字典变更不会同步），ChainMap只是对原来字典的引用（原字典变更会同步）
# 字符串处理
# 分隔字符串
line = 'asdf fjdk; afed, fjek,asdf, foo'
import re
re.split(r'[;,\s]\s*', line)
# ['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']
# 也可以使用一个括号捕获分组，使用了捕获分组，那么被匹配的文本也会出现在列表中
fields = re.split(r'(;|,|\s)\s*', line)
# ['asdf', ' ', 'fjdk', ';', 'afed', ',', 'fjek', ',', 'asdf', ',', 'foo']
# 可以通过捕获后获取分隔字符串，从而构造出新的字符串等
values = fields[::2]
# ['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']
delimiters = fields[1::2]
# [' ', ';', ',', ',', ',', '']
''.join(v+d for v,d in zip(values, delimiters))
# 'asdf fjdk;afed,fjek,asdf,foo'
# 如果你希望通过括号来分组但是不希望捕获，则可以通过(?:...)
re.split(r'(?:,|;|\s)\s*', line)
# ['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']

# startswith() 或者 endswith() 方法：
import os
filenames = os.listdir('.')
# filenames [ 'Makefile', 'foo.c', 'bar.py', 'spam.c', 'spam.h' ]
# match 多个
[name for name in filenames if name.endswith('.c', '.h')]
# any满足
any(name.endswith('.py') for name in filenames)

# 另一个例子
from urllib.request import urlopen

def read_data(name):
    # 利用元组来匹配多个
    if name.startswith(('http:', 'https:', 'ftp:')):
        return urlopen(name).read()
    else:
        # 否则，读取文件
        with open(name) as f:
            return f.read()
# startswith必须使用元组作为参数，如果参数不是元组类型，需使用tuple将其转换为元组类型
choices = ['http:', 'ftp:']
url = 'http://www.python.org'
url.startswith(choices) # 报错
url.startswith(tuple(choices)) # 可正常执行
# 也可以采用切片的方式去获取
filename = 'spabm.txt'
filename[-4:] == '.txt'

import re
url = 'http://www.python.org'
re.match('http:|https:|ftp:', url)
# startswith 和 endswith可以很方便的用来匹配文件后缀名

# 使用通配符或者正则来匹配文本字符串
# fnmatch 使用底层操作系统的大小写敏感规则（不同的系统不一样）来匹配
from fnmatch import fnmatch, fnmatchcase
fnmatch('foo.txt', '*.txt') # true
fnmatch('foo.txt', '?oo.txt') # true
fnmatch('Dat45.csv', 'Dat[0-9]*') # true
# fnmatchcase 则完全根据指定模式匹配
fnmatchcase('foo.txt', '*.TXT')
# 也可以用于处理非文件名
addresses = [
    '5412 N CLARK ST',
    '1060 W ADDISON ST',
    '1039 W GRANVILLE AVE',
    '2122 N CLARK ST',
    '4802 N BROADWAY',
]
[addr for addr in addresses if fnmatchcase(addr, '* ST')]
[addr for addr in addresses if fnmatchcase(addr, '54[0-9][0-9] *CLARK*')]
# fnmatch的能力很适合使用正则表达式和通配符匹配字符串的场景，如果需要做文件名的匹配，最好使用glob模块
# 可使用fins查找目标字符串的索引
text = 'yeah, but no, but yeah, but no, but yeah'
text.find('no') # 10
# 正则匹配
import re
if re.match(r'\d+/\d+/\d+', text):
# 匹配成功
    print('yes')
else:
    print('text does not match')
# 匹配失败

# 如果你想使用同一个模式去做多次匹配，你应该先将模式字符串编译为模式对象
datepat = re.compiler(r'\d+/\d+/\d+')
# if datepat.match(text1):
#     print('yes')
# else:
#     print('no')
# if datepat.match(text2):
#     print('yes')
# else:
#     print('no')    

# match总是从字符串开始去匹配，如果你想查找字符串任意部分的模式出现为止，使用findall方法去替代
text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
datepat.findall(text)   
# 在定义正则的时候，通常会用括号去捕获分组
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')
m = datepat.match('11/27/2012')
# extract the contents of each group
m.group(0) # '11/27/2012'
m.group(1) # '11'
m.group(2) # '27'
m.group(3) # '2012'
m.groups() # ('11', '27', '2012')
month, day, year = m.groups()
# 也可以使用findall方法去捕获分组
datepat.findall(text)
# [('11', '27', '2012'), ('3', '13', '2013')]
# 也可以使用finditer方法去捕获
text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
for month, day, year in datepat.findall(text):
    print('{}-{}-{}'.format(year, month, day))

# findall方法会以列表形式发挥匹配，可以使用finditer方法来以迭代方式返回匹配
for m in datepat.finditer(text):
    print(m.groups())
# ('11', '27', '2012')
# ('3', '13', '2013')   
# match只匹配字符串的开始部分，它的匹配结果可能不符合预期，可以使用$结尾来处理
# 如果仅仅是做一次简单的文本/搜索操作的话，可以略过编译部分，直接使用re模块级别的函数
re.findall(r'(\d+)/(\d+)/(\d+)', text)
# [('11', '27', '2012'), ('3', '13', '2013')]
# 对简单的模式，使用replace即可，复杂的模式使用re.sub
text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
import re
# 被匹配的模式 替换模式
# 如果要支持多次替换，考虑先编译它来提升性能
re.sub(r'(\d+)/(\d+)/(\d+)', r'\3-\1-\2-', text)
# 'Today is 2012-11-27. PyCon starts 2013-3-13.'
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')
datepat.sub(r'\3-\1-\2-', text)
# 'Today is 2012-11-27. PyCon starts 2013-3-13.'

# 正则表达式的元字符
# 如果你使用了明明分组，那么第二个参数请使用\g<group_name>
re.sub(r'(?P<month>\d+)/(?P<day>\d+)/(?P<year>\d+)', r'\g<year>-\g<month>-\g<day>', text)
# 'Today is 2012-11-27. PyCon starts 2013-3-13.'
# 对于复杂的替换，可以传递一个替换函数来代替
from calendar import month_abbr
def change_date(m):
    mon_name = month_abbr[int(m.group[1])]
    return '{} {} {}'.format(m.group[2], mon_name, m.group(3))
datepat.sub(change_date, text)
# 'Today is 27 November 2012. PyCon starts 13 March 2013.'
# 如果想知道有多少替换发生了，可以使用subn
newtext, n = datepat.subn(r'\3-\1-\2', text)
# 3
# 匹配时忽略大小写
text = 'UPPER PYTHON, lower python, Mixed Python'
re.findall('python', text, flags=re.IGNORECASE)
# ['PYTHON', 'python', 'Python']
re.sub('python', 'snake', text, flags=re.IGNORECASE)   
# 'UPPER snake, lower snake, Mixed snake'
# 替换字符与原本字符的大小写保持一致
def matchcase(word):
    def replace(m):
        text = m.group()
        if text.isupper():
            return word.upper()
        elif text.islower():
            return word.lower()
        elif text[0].isupper():
            return word.capitalize()
        else:
            return word
    return replace

re.sub('python', matchcase('snake'), text, flags=re.IGNORECASE)
# 'UPPER SNAKE, lower snake, Mixed snake'
# 简单的忽略大小写用e.IGNORECASE就够了，但是对于某些需要大小转换的unicode匹配可能还不够
# 默认的匹配模式是贪婪匹配，会返回最长的可能匹配，可以通过在操作服后面加上？修饰符，变成非贪婪模式
str_pat = re.compile(r'"(.*)"')
text1 = 'Computer says "no."'
str_pat.findall(text1)
# ['no.']
text2 = 'Computer says "no." Phone says "yes."'
str_pat.findall(text2)
# ['no." Phone says "yes.']
# 非贪婪匹配,?可以强制匹配算法改成寻找最短的可能匹配
str_pat = re.compile(r'"(.*?)"')
str_pat.findall(text2)
# ['no.', 'yes.']
# re.DOTALL可以让(.)匹配包含换行符在内的任意字符
comment = re.compiler(r'/\*(.*?)\*', re.DOTALL)
text2 = '''/* this is a
 multiline comment */'''
comment.findall(text2)
# [' this is a\n multiline comment ']
# 如果只是简单的匹配换行，使用re.DOTALL即可，如果复杂的模式匹配还是定义自己的表达式模式比较好
# 在需要比较字符串的程序中使用字符的多种表示会产生问题，你可以使用unicodedata模块将文本标准化
s1 = 'Spicy Jalape\u00f1o'
s2 = 'Spicy Jalapen\u0303o'
s1 == s2
# False
import unicodedata
t1 = unicodedata.normalize('NFC', s1)
t2 = unicodedata.normalize('NFC', s2)
t1 == t2
# True
t1 = unicodedata.normalize('NFD', s1)
t2 = unicodedata.normalize('NFD', s2)
# NFD 和 NFC指的是字符标准化的方式
# 比较、清理、过滤文本时，字符的标准化是很重要的
t1 = unicodedata.normalize('NFD', s1)
# 过滤s1中的变音符 ，combining 判断字符是否为和音字符
''.join(c for c in t1 not unicodedata.combining(c))
# 'Spicy Jalapen\u0303o'
# 匹配unicode字符
arabic = re.compile('[\u0600-\u06ff\u0750-\u088f\u08a0-\u08ff]+')
# 当执行匹配和搜索的时候，最好先标准化且清理所有文本为标准化格式
# 需要注意一些特殊情况，比如忽略大小写匹配和大小写转换时的行为
pat = re.compile('stra\u00dfe', re.IGNORECASE)
s = 'straße'
pat.match(s)
# <_sre.SRE_Match object at 0x10069d370>
pat.match(s.super()) # Doesn't match
# 混合使用unicode和正则表达式很痛苦，可以考虑安装第三方正则库，它们会为unicode的大小写转换和其他有趣特性提供全面的支持，包括模糊匹配

# strip去除字符串前后的空格和换行， 中间的字符可以用replace 或者sub，可以与迭代操作结合
#  lines = (line.strip() for line in f) 执行数据转换操作会非常高效，因为不需要预先读取所有数据放到一个临时的列表中去，仅仅创建一个生成器
with open(filename) as f:
    lines = (line.strip() for line in f)
    for line in lines:
        print(line)
# 创建一个小的转换表格然后使用translate方法
# 简单字符的映射或删除，可直接使用translate，复杂字符操作可以使用translate
        
remap = {
    ord('\t') : ' ',
    ord('\f') : ' ',
    ord('\r') : None, # deleted
}
a = s.translate(remap)

# 删除所有的和音符
import unicodedata
import sys
# 构造一个字典，每个unicode和音符作为键，对应的值全部为none
cmb_chrs = dict.fromkeys(c for c in range(sys.maxunicode)
    # 将原始输入标准化为分解形式字符，调用translate删除所有重音符
    if unicodedata.combining(chr(c)))
b = unicodedata.normalize('NFD', a)
b.translate(cmb_chrs)
# 构造一个所有unicode字符映射到ascii字符上的表格
digitmap = {c: ord('0') + unicodedata.digit(chr(c))
    for c in range(sys.maxunicode)
    if unicodedata.category(chr(c) == 'Nd')}
len(digitmap)
x = '\u0661\u0662\u0663'
x.translate(digitmap)
b = unicodedata.normalize('NFD', a)
b.encode('ascii', 'ignore').decode('ascii')
# 字符串对齐的方法
text = 'hello world'
text.ljust(20, '=')
# '=========Hello World'
text.center(20, '*')
# '****Hello World*****'
# format可以很容易的对齐字符串
format(text, '>20')
format(text, '=>20s')
format(text, '*<20s')
# 格式化多个值
'{:>10s} {:>10s}'.format('Hello', 'World')
# format可以用来格式化任何值，是格式化不同类型的值的好选择
x = 1.2345
format(x, '^10.2f')

# 两个字符串合并，则不需要加号，只需要放到一起
a = 'hello' 'world'
# helloworld
# +连接大量字符串是非常低效的，因为+连接回忆起内存复制以及垃圾回收操作，使用join连接更快
# 如果可以，最好先收集起来然后join，不行的话，使用生成器表达式也可以
data = ['ACME', 50, 91.1]
','.join(str(d) for d in data)
# 'ACME,50,91.1'

# 注意不要做不必要的连接操作
print(a + ':' + b + ":" + c) #ugly
print(':'.join([a, b, c])) #ugly
print(a, b, c, sep=":") #ugly

# 你可以使用生成器函数来构建大量小字符串的输出代码
def sample():
    yield 'Is'
    yield 'Chicago'
    yield 'Not'
    yield 'Chicago?'

def combine(source, maxsize):
    parts = []
    size = 0
    for part in source:
        parts.append(part)
        size += len(part)
        if size > maxsize:
            yield ''.join(parts)
            parts = []
            size = 0
    yield ''.join(parts)    
# 结合文件操作
with open('filename', 'w') as f:
    for part in combine(sample(), 32768):
        f.write(part)

# 使用format替换字符串中的变量
s = '{name} has {n} messages.'
s.format(name="gyu", n=37)
# 'Guido has 37 messages.'     
   
name = 'Guido'
n = 37
s.format_map(vars())
# 'Guido has 37 messages.'
# vars 也适用于对象实例
class Info:
    def __init__(self, name, n):
        self.name = name
        self.n = n
a = Info('hhh', 37)
# 若变量缺失会报错
s.format_map(vars(a))
# 'hhh has 37 messages.'

# 变量缺失则直接返回
s = '{name} has {n} messages.'
class safesub(dict):
    def __missing__(self, key):
        return '{'+ key + '}'
s.format_map(safesub(vars()))
# 'Guido has {n} messages.'

# 如果频繁执行则考虑封装
import sys
def sub(text):
    # sys._getframe(1).f_locals 和 vars 有同样的功能
    # sys._getframe(1) 返回调用者的栈帧，f_locals获得局部变量，f_locals是本地变量的复制，对他的操作不会覆盖和改变调用者本地变量的值
    # 多数情况下直接操作栈帧是不推荐的，注意
    return text.format_map(safesub(sys._getframe(1).f_locals)
                           

import os
os.get_terminal_size().columns
import textwrap
s = "Look into my eyes, look into my eyes, the eyes, the eyes, \
the eyes, not around the eyes, don't look around the eyes, \
look into my eyes, you're under."
# textwrap对于字符串的打印是非常有用的，当你希望输出自动匹配终端大小的时候
# 你可以使用os.get_terminal_size 方法来获取终端的大小尺寸.
# fill方法接受一些其他的可选参数来控制tab，语句结尾等
print(textwrap.fill(s, 40, initial_indent = '   #', subsequent_indent='      _')) 
# 有一些工具函数可以帮助你处理xml或html  
# 利用命名捕获组的正则表达式来定义所有可能的令牌
import re
NAME = r'(?P<NAME>[a-zA-Z_][a-zA-Z_0-9]*)'
NUM = r'(?P<NUM>\d+)'
PLUS = r'(?P<PLUS>\+)'
TIMES = r'(?P<TIMES>\*)'
EQ = r'(?P<EQ>=)'
WS = r'(?P<WS>\s+)'

master_pat = re.compile('|'.join([NAME, NUM, PLUS, TIMES, EQ, WS]))
# 借助于scanner方法，不断调用match，可以获得每一个匹配的目标文本
# 可以将处理令牌的逻辑打包到一个生成器中
def generate_tokens(pat, text):
    Token = namedtuple('Token', ['type', 'value'])
    scanner = pat.scanner(text)
    for m in iter(scanner.match, None):
        yield Token(m.lastgroup, m.group())
# example use
for tok in generate_tokens(master_pat, 'foo=42')
    print(tok)
# 过滤令牌流
tokens = (tok for tok in generate_tokens(master_pat, text) if tok.type !== 'WS')
# 通常来讲令牌化湿很多高级文本解析与处理的第一步，但是使用时你需要确认正则表达式指定了所有输入中可能出现的文本序列
# 如果有任何不可匹配的文本出现了，扫描就会停止
# 令牌的顺序是有影响的，re模块会按照指定好的顺序去做匹配
# 如果一个模式恰好是另一个更长模式的子字符串，那么你需要确定长模式写在前面
# 构建一个递归下降表达式求值程序
import re
import collections

# Token specification
NUM = r'(?P<NUM>\d+)'
PLUS = r'(?P<PLUS>\+)'
MINUS = r'(?P<MINUS>-)'
TIMES = r'(?P<TIMES>\*)'
DIVIDE = r'(?P<DIVIDE>/)'
LPAREN = r'(?P<LPAREN>\()'
RPAREN = r'(?P<RPAREN>\))'
WS = r'(?P<WS>\s+)'

master_pat = re.compile('|'.join([NUM, PLUS, MINUS, TIMES,
                                  DIVIDE, LPAREN, RPAREN, WS]));
# tokenizer
Token = collections.namedtuple('Token', ['type', 'value'])

def generate_tokens(text):
    scanner = master_pat.scanner(text)
    for m in iter(scanner.match, None):
        tok Token(m.lastgroup, m.group())
        if tok.type != 'WS':
            yield tok

# parser
class ExpressionEvaluator:
 '''
    Implementation of a recursive descent parser. Each method
    implements a single grammar rule. Use the ._accept() method
    to test and accept the current lookahead token. Use the ._expect()
    method to exactly match and discard the next token on on the input
    (or raise a SyntaxError if it doesn't match).
    '''
    def parse(self, text):
        self.tokens = generate_tokens(text)
        self.tok = None  # Last symbol consumed
        self.nexttok = None  # Next symbol tokenized
        self._advance()  # Load first lookahead token
        return self.expr()

    def _advance(self):
        'Advance one token ahead'
        self.tok, self.nexttok = self.nexttok, next(self.tokens, None)

    def _accept(self, toktype):
        'Test and consume the next token if it matches toktype'
        if self.nexttok and self.nexttok.type==toktype:
            self._advance()
            return True
        return False
    def _expect(self, toktype):
       'Consume next token if it matches toktype or raise SyntaxError'
        if not self._accept(toktype):
            raise SyntaxError('Expected'+ toktype)
    # Grammer rules follow
    def expr(self):
        "expression ::= term { ('+'|'-') term }*"
        exprval = self.term()
        while self._accept('PLUS') or self._accept('MINUS'):
            op = self.tok.type
            right = self.term()
            if op == 'PLUS':
                exprval += right
            elif op == 'MINUS':
                exprval -= right
        return exprval
    
    def term(self):
        "term ::= factor { ('*'|'/') factor }*"
        itemval = self.factor()
        while self._accept('TIMES') or self._accept('DIVIDE'):
            op = self.tok.type
            right = self.factor()
            if op == 'TIMES':
                itemval *= right
            elif op == 'DIVIDE':
                itemval /= right
        return itemval

def factor(self):
    "factor ::= NUM | ( expr )"
    if self._accept('NUM'):
        return int(self.tok.value)
    elif self._accept('LPAREN'):
        exprval = self.expr()
        self._expect(RPAREN)
        return exprval
    else:
        raise SyntaxError('Expected NUMBER or LPAREN')

def descent_parser():
    e = ExpressionEvaluator()
    print(e.parse('2'))
    print(e.parse('2 + 3'))
    print(e.parse('2 + 3 * 4'))
    print(e.parse('2 + (3 + 4) * 5'))
        # print(e.parse('2 + (3 + * 4)'))
    # Traceback (most recent call last):
    #    File "<stdin>", line 1, in <module>
    #    File "exprparse.py", line 40, in parse
    #    return self.expr()
    #    File "exprparse.py", line 67, in expr
    #    right = self.term()
    #    File "exprparse.py", line 77, in term
    #    termval = self.factor()
    #    File "exprparse.py", line 93, in factor
    #    exprval = self.expr()
    #    File "exprparse.py", line 67, in expr
    #    right = self.term()
    #    File "exprparse.py", line 77, in term
    #    termval = self.factor()
    #    File "exprparse.py", line 97, in factor
    #    raise SyntaxError("Expected NUMBER or LPAREN")
    #    SyntaxError: Expected NUMBER or LPAREN
if __name__ == '__main__':
    descent_parser()

# 字符串操作同样适用于字节数组
data = bytearray(b'hello world')
print(data[0:5])
data.startswith(b'hello')
data.split()
data.replace(b'hello', b'hello cruel')

# 你可以用正则表达式匹配字符串，但是正则表达式本身必须也是字符串
data = b'FOO:BAR,SPAM'
import re
re.split('[:,]', data)
re.split(b'[:,]', data)

# 大多数情况下，在文本字符串上的操作均可用于字节字符串，然后，也有一些不同
# 首先，字节字符串的索引操作返回整数而不是单独字符
a = 'Hello World'
a[0] # 'H'
a[1] # 'e'

b = b'Hello World'
b[0] # 72
b[1] # 101

# 字节字符串不能很好的打印出来
s = b'Hello World'
print(s)
# b'Hello World' # Observe b'...'
print(s.decode('ascii'))
# Hello World

# 字节字符串也不支持类似%之类的格式化操作

# 如果想格式化字节字符串，要先使用标准的文本字符串，然后将其编码为字节字符串
'{:10s} {:10d} {:10.2f}'.format('ACME', 100, 490.1).encode('ascii')
# b'ACME 100 490.10'
# 使用字节字符串可能会改变一些操作的语义，特别是那些跟文件系统有关的操作
# 如果你使用一个编码为字节的文件名，而不是一个普通的我呢吧呢字符串，会禁用文件名的编码/解码
with open('jalape\xf1o.txt', 'w') as f:
    f.write('spicy')

import os
os.listdir('.')
# ['jalapeño.txt']
os.listdir(b'.')
# [b'jalapen\xcc\x83o.txt']
# 通常来说，使用字节字符串处理文本更加高效，但是字节字符串并不能和puthon的其他部分工作的很好，你还需要手动处理所有的编码/解码操作
# 处理文本时，还是建议直接选择普通的文本字符串而不是字节字符串
 