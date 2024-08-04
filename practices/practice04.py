# 读写csv数据为一个元组
import csv

with open('stocks.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    # 默认情况下使用索引访问row的指定列
    for row in f_csv:
        # process row

# 使用命名元组
from collections import namedtuple
col_types = [str, float, str, str, float, int]

with open('stocks.csv') as f:
    f_csv = csv.reader(f)
    # 处理标题中的非法标识符
    headers = [re.sub('[^a-zA_Z]', '_', h) for h in next(f_csv)]
    headings = next(f_csv)
    Row = namedtuple('Row', headings)
    for r in f_csv:
        # process row
        row = Row(*r)
        print(row.Symbol)
        print(row.Change)
        # 额外处理csv产生的数据，做一些类型转换
        row = tuple(convert(value) for convert, value in zip(col_types, row))
        # ...
# 使用下标访问需要列名是合法的python字符，如果不是的话，需要做额外处理
# 另外一个选择就是将数据读取到一个字典序列化中去
import csv
with open('stocks.csv') as f:
    f_csv = csv.DictReader(f)
    headings = next(f_csv)
    for r in f_csv:
       # process row
       print(row['Symbol'])
       print(row['Change']) 

# 为了写入csv数据，你可以使用csv模块，先创建一个writer对象
headers = ['Symbol','Price','Date','Time','Change','Volume']
rows = [('AA', 39.48, '6/11/2007', '9:36am', -0.18, 181800),
         ('AIG', 71.38, '6/11/2007', '9:36am', -0.15, 195500),
         ('AXP', 62.58, '6/11/2007', '9:36am', -0.46, 935000),
       ]
with open('stocks.csv', 'w') as f:
    f_csv = csv.writer(f)
    # 写入标题列
    f_csv.writerow(headers)
    # 写入每一行的数据
    f_csv.writerows(rows)

# 如果你有一个字典序列的数据，可以这样做


headers = ['Symbol', 'Price', 'Date', 'Time', 'Change', 'Volume']
rows = [{'Symbol':'AA', 'Price':39.48, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.18, 'Volume':181800},
        {'Symbol':'AIG', 'Price': 71.38, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.15, 'Volume': 195500},
        {'Symbol':'AXP', 'Price': 62.58, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.46, 'Volume': 935000},
        ]
with open('stocks.csv', 'w') as f:
    f_csv = csv.DictWriter(f)
    # 写入标题列
    f_csv.writeheader(headers)
    # 写入每一行的数据
    f_csv.writerows(rows)

# 使用csv库可以识别csv编码规则，然后会带来很好的兼容性
# 读取tab分割的数据
with open('stock.tsv') as f:
    t_tsv = csv.reader(f, delimiter='\t')
    for row in t_tsv:
        # Process row

# 转换字典中特定字段
field_types = [('Price', float), ('Change', float), ('Volume', int)]
with open('stocks.csv') as f:
    for row in csv.DictReader(f):
        row.update((key, conversion(row[key])) for key, conversion in field_types)
        print(row)
# 读取csv数据做数据分析和统计的话，pandas包是一个很好的选择，pandas.read_csv(),可加载csv数据到一个dataframe对象中去，然后利用这个对象你就可以生成各种形式的统计，过滤数据以及执行其他高级操作

# 空间补丁使得模型可以更细致的处理视频内容的每一小段，同时考虑它们随时间的变化
# 空间时间补丁首先通过压缩网络生成，这一网络负责将原始视频数据压缩成更低维度的表示形式，即一个由许多小块组成的密集网络。
# 这些小块即为我们所说的补丁，每个补丁都携带了一部分视频的空间和时间信息
# 有了这些空间时间补丁，sora就可以开始他们的转换过程了，通过预先训练好的转换器（Transformer模型），sora能够识别每个补丁的内容，并根据给定的文本提示进行相应的修改
# 基于空间补丁的处理方式允许sora以非常精细的层次操作视频内容，因为它可以独立处理视频中的每一块信息，其次这种方式极大的提高了处理视频的灵活性，使得sora能够生成剧透复杂动态的高质量视频
# 此外，通过对这些补丁进行有效管理和转换，sora能够在保证视频内容连贯性的同时，创造出丰富的视觉效果，满足用户的各种需求
# 空间补丁是sora能够生成高质量视频的关键因素之一
# 视频生成过程的三个关键步骤：视频压缩网络、空间时间潜在补丁提取、视频生成的transformer模型
# 视频压缩网络：将一段视频的内容“打扫和组织”成一个更加紧凑、高效的形式（即降维），这样sora就能子啊处理时更高效，同时仍保留足够的信息来重建原始视频
# 空间时间潜在补丁提取：通过视频压缩网络处理后，sora会将视频分解成一个个小块，这些小块包含视频的一小部分的空间和时间信息，这使得sora在之后的步骤能鼓针对性的提取和处理视频的每一部分
# 视频生成的transformer模型：在sora的视频生成过程中，transformer模型正扮演着类似的角色，它接收空间时间潜在补丁（即视频内容的“拼图切块”）和文本提示（即故事），然后决定如何将这些片段转换或组合成最终的视频，从而描述文本提示中的故事。
# 通过上述三个步骤的协同工作，sora能够将文本提示转化为具有丰富细节和动态效果的视频内容，极大的提升了视频内容生成的灵活性和创造力
# sora对三维空间理解的深度
# sora在维护视频的3D一致性和长期一致性方面的能力都很强大，sora还能模拟真实世界的互动，比如绘画时在画布上留下痕迹，痕迹随时间的推移而累积
# sora也存在一定的局限性，在模拟物理世界的准确性存在局限，对于复杂的物理互动，如玻璃破碎的惊喜过程，或是设计精确力学的运动场景，主要是因为训练数据中缺乏足够的实例来让模型学写这些复杂的物理现象
# 克服物理世界模拟的局限性的策略：
# - 扩大训练数据集：集成更多包含复杂物理互动的高质量视频数据，以丰富sora学习的样本
# - 物理引擎集成：在sora框架中集成物理引擎，让模型在生成视频时能参考物理规则，提高物理互动的真实性
# 生成长视频的困难克服的策略：
# - 增加时间连续性学习：通过改进训练算法，增强模型对时间连续性和逻辑一致性的学习能力
# - 在视频生成过程中，采取序列化处理的方法，按照时间顺序逐帧生成视频 ，保证每一帧的前后一致性
# 准确理解复杂文本指令的问题克服策略：
# - 改善语言模型：提升sora内嵌的语言理解模型的复杂度与准确性，使其能够更好的理解和分析复杂文本指令
# - 文本预处理：引入先进的文本预处理步骤，将复杂的文本指令分解为简单的、易于理解模型的多个字任务，逐一生成，最后综合为完整视频
# 训练与生成效率的客服策略：
# - 优化模型结构：对sora的架构进行优化，减少不必要的计算，提高运行效率
# - 硬件加速：利用更强大的计算资源和专门的硬件加速技术，缩短视频生成的时间
        
# 漂亮的打印twitter上搜索结果的例子
# pprint 会按照key的字母顺序并以一种更加美观的方式输出
from urllib.request import urlopen
import json
u = urlopen('http://search.twitter.com/search.json?q=python&rpp=5')
resp = json.load(u.read().decode('utf-8'))
from pprint import pprint
pprint(resp)
# 解码json数据并在orderedDict中保留其顺序的例子
s = '{"name": "ACME", "shares": 50, "price": 490.1}'
from collections import OrderedDict
data = json.loads(s, object_pairs_hook=OrderedDict)
# OrderedDict([('name', 'ACME'), ('shares', 50), ('price', 490.1)])
# 将一个json字典转换为一个python对象例子
class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

data = json.loads(s, object_hook=JSONObject)
# data.name # 'ACME'
# data.shares # 50
# data.price # 490.1
# 格式化的输出编码后的数据
print(json.dumps(data, indent=4))
# {
#     "price": 542.23,
#     "name": "ACME",
#     "shares": 100
# }

# 对象实例通常并不是json可序列化的
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
p = Point(2, 3)
json.dumps(p)
# 会报错，如果想序列化对象实例，可以提供一个函数，它的输入是一个实例，返回一个可序列化的字典
def serialize_instance(obj):
    d = {'__classname__': type(obj).__name__ }
    d.update(vars(obj))
    return d

# dictionary mapping names to known classes
classes = {
    'Point': Point
}
def unserialize_object(d):
    clsname = d.pop('__classname__', None)
    if clsname:
        cls = classes[clsname]
        obj =cls.__new__(cls) # make instance without calling __init__
        for key, value in d.items():
            setattr(obj, key, value)
        return obj
    else:
        return d

p = Point(2,3)
s = json.dumps(p, default=serialize_instance)
# '{"__classname__": "Point", "y": 3, "x": 2}'

a = json.loads(s, object_hook=unserialize_object)
# a是一个puthon的对象实例
# <__main__.Point object at 0x1017577d0>
a.x # 2
a.y # 3

# 从一个简单的xml文档中提取数据
from urllib.request import urlopen
from xml.etree.ElementTree import parse

# download the rss feed and parse it
u = urlopen('http://planet.python.org/rss20.xml')
doc = parse(u)

# extract and output tags of interest
for item in doc.iterfind('channel/item'):
    title = item.findtext('title')
    date = item.findtext('pubDate')
    link = item.findtext('link')

    print(title)
    print(date)
    print(link)

# doc elementTree
e = doc.find('channel/title')
e.tag
e.text
e.get('some_attribute')
# 任何时候只要你遇到增量式的处理数据时，第一时间就应该想到迭代器和生成器
from xml.etree.ElementTree import iterparse

def parse_and_remove(filename, path):
    path_parts = path.split('/')
    # 允许对xml文档进行增量操作
    doc = iterparse(filename, ('start', 'end'))
    # skip the root element
    next(doc)

    tag_stack = []
    elem_stack = []
    for event, elem in doc:
        # start和end事件被用来管理元素和标签栈
        if event == 'start':
            tag_stack.append(elem.tag)
            elem_stack.append(elem)
        elif event == 'end':
            if tag_stack == path_parts:
                yield elem
                # yield之后产生的元素从它的父节点中删除掉，会触发内存回收
                elem_stack[-2].remove(elem)
            try:
                tag_stack.pop()
                elem_stack.pop()
            except IndexError:
                pass

# 按照坑洼报告数量排列邮编号码
from xml.etree.ElementTree import parse
from collections import Counter

potholes_by_zip = Counter()

doc = parse('potholes.xml')
for pothole in doc.iterfind('row/row'):
    potholes_by_zip[pothole.findtext('zip')] += 1
for zipcode, num in potholes_by_zip.most_common():
    print(zipcode, num)
# 上述代码会将整个XML文件加载到内存中然后解析，需要450MB的内存，下面修改后的代码只需要7MB的内存
from collections import Counter
potholes_by_zip = Counter()

# 分批次加载？？？
data = parse_and_remove('potholes.xml', 'row/row')
for pothole in data:
    potholes_by_zip[pothole.findtext('zip')] += 1
# 内存占用的减少也会带来处理速度的下降，因为不是一次性将文件加载到内存中的，所以会存在多次加载文件内容到内存的耗时
for zipcode, num in potholes_by_zip.most_common():
    print(zipcode, num)
# python字典存储数据转换成xml格式
from xml.etree.ElementTree import Element

def dict_to_xml(tag, d):
    '''
    Turn a simple dict of key/value pairs into XML
    '''
    elem = Element(tag)
    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)
    return elem
s = { 'name': 'GOOG', 'shares': 100, 'price':490.1 }
e = dict_to_xml('stock', s)
e
# <Element 'stock' at 0x1004b64c8>
# e是一个Element实例，使用tostring方法可以将其转换成一个字节字符串
from xml.etree.ElementTree import tostring
tostring(e)
# b'<stock><price>490.1</price><shares>100</shares><name>GOOG</name></stock>'

# 给某个元素添加属性值，使用set方法
e.set('_id', '1234')
tostring(e)
# b'<stock _id="1234"><price>490.1</price><shares>100</shares><name>GOOG</name>
# </stock>'
# 如果想保持创建出来的xml标签的顺序，可以考虑构造一个OrderedDict来代替普通的字典
# 当你创建xml的时候，你被限制只能构造字符串类型的值
# 读取xml文档，做一些修改后写入
from xml.etree.ElementTree import parse, Element
doc = parse('pred.xml')
root = doc.getroot()
root
# <Element 'stop' at 0x100770cb0>
# remove a few elements
root.remove(root.find('sri'))
root.remove(root.find('cr'))
# Insert a new element after <nm>...</nm>
root.getchildren().index(root.find('nm')) # 1
e = Element('spam')
e.text = 'this is a test'
root.insert(2, e)
# write back to a file
doc.write('newpred.xml', xml_declaration=True)
# 对xml元素的操作总是通过父节点，可通过element[i]或element[i:] 对元素使用索引和切片操作
# 对一个使用了命名空间的xml文档执行普通的查询会很繁琐（需要添加命名空间的前缀）
# 你可以通过将命名空间包装为一个工具来简化这个过程
class XMLNamespaces:
    def __init__(self, **kwargs):
        self.namespaces = {}
        for name, uri in kwargs.items():
            self.register(name, uri)
            print(kwargs.items(), 'kwargs.items()----')
            print(self.namespace, 'self.namespace----')
    def register(self, name, uri):
        self.namespaces[name] = '{'+uri+'}'
    def __call__(self,path):
        return path.format_map(self.namespaces)
    
ns = XMLNamespaces(html='http://www.w3.org/1999/xhtml')
# 以变量插槽的方式书写，借助XMLNamespaces方法完成变量名-命名空间的替换
doc.find(ns('content/{html}html'))
# <Element '{http://www.w3.org/1999/xhtml}html' at 0x1007767e0>
doc.findtext(ns('content/{html}html/{html}head/{html}title'))
# 'Hello World'

# 在基本的elementTree解析中没有任何途径获取命名空间的信息，但是，如果你使用iterparse函数就可以获取更多关于命名空间处理范围的信息
from xml.etree.ElementTree import iterparse
for evt,elem in iterparse('ns2.xml', ['end', 'start-ns', 'end-ns']):
    print(evt, elem)
# end <Element 'author' at 0x10110de10>
# start-ns ('', 'http://www.w3.org/1999/xhtml')
# end <Element '{http://www.w3.org/1999/xhtml}title' at 0x1011131b0>
# end <Element '{http://www.w3.org/1999/xhtml}head' at 0x1011130a8>
# end <Element '{http://www.w3.org/1999/xhtml}h1' at 0x101113310>
# end <Element '{http://www.w3.org/1999/xhtml}body' at 0x101113260>
# end <Element '{http://www.w3.org/1999/xhtml}html' at 0x10110df70>
# end-ns None
# end <Element 'content' at 0x10110de68>
# 如果你处理的xml文件除了要使用其他高级的xml特性外，还使用到命名空间，建议你最好使用lxml函数库来代替elementtree，lxml对利用dtd验证文档，
# 更好的xpath支持和其他一些高级xml特性等提供了更好的支持
# python表示多行数据的标准方式是一个由元组构成的序列

# 操作关系型数据库
import sqlite3
db = sqlite3.connect('database.db')
c = db.cursor()
c.execute('create table portfolio (symbol text, shares integer, price real)')
# <sqlite3.Cursor object at 0x10067a730>
db.commit()

stocks = [
    ('GOOG', 100, 490.1),
    ('AAPL', 50, 545.75),
    ('FB', 150, 7.45),
    ('HPQ', 75, 33.2),
]

c.executemany('insert into portfolio values (?,?,?)', stocks)
db.commit()

# 接受用户输入作为参数来查询
min_price = 100
for row in db.execute('select * from portfolio where price >= ?', (min_price, )):
    print(row)

# 操作数据库需要注意：
# - 不同数据类型的数据处理
# - 防SQL注入攻击
# 编码和解码十六进制数
s = b'hello'
import binascii

h = binascii.b2a_hex(s)
# b'68656c6c6f' 总是输出字节字符串，如果想输出unicode形式的字符串，需要一个额外的转换操作
binascii.a2b_hex(h)
# b'hello'
s = b'hello'
import base64
a = base64.b84decode(s)
# b'aGVsbG8='
base64.b64decode(a)
# b'hello'

# base64编码仅仅用于面向字节的数据比如字节字符串或字节数组，输出结果总是一个字节字符串，
# 如果想调整输出unicode字符，则需decode一下
base64.b64encode(a).decode('ascii') # 'aGVsbG8='
# 当解码base64的时候，字节字符串和unicode文本都可以作为参数，但是unicode字符串只能包含ascii字符

# struct模块处理二进制数据
# 将一个python元组列表写入一个二进制文件，并使用struct将每个元组编码为一个结构体
from struct import Struct
def write_records(records, format, f):
    '''
    Write a sequence of tuples to a binary file of structures.
    '''
    record_struct = Struct(format)
    for r in records:
        r.write(record_struct.pack(*r))
    
# Example
if __name__ == '__main__':
    records = [ (1, 2.3, 4.5),
                (6, 7.8, 9.0),
                (12, 13.4, 56.7) ]
    with open('data.b', 'wb') as f:
        write_records(records, '<idd', f)
# 以块的形式增量读取文件并返回元组列表
from struct import Struct

def read_records(format, f):
    record_struct = Struct(format)
    chunks = iter(lambda: f.read(record_struct.size), 'b')
    return (record_struct.unpack(chunk) for chunk in chunks)

if __name__ == '__main__':
    with open('data.b', 'rb') as f:
        for rec in read_records('<idd', f):
            # process rec
# 将文件一次性读取到一个字节字符串中，然后分片解析
from struct import Struct

def unpack_records(format, data):
    record_struct = Struct(format)
    return (record_struct.unpack_from(data, offset)
            for offset in range(0, len(data), record_struct.size))

if __name__ == '__main__':
    with open('data.b', 'rb') as f:
        data = f.read()
    for rec in unpack_records('<idd', f):
        # process rec
# 通常创建一个迭代器iter+迭代条件 可以同来代替循环
# 如果你的程序需要处理大量的二进制数据，你最好使用numpy模块
# 它可以j将数据读取到一个结构化数组而不是以爱过元组列表中
import numpy as np
f = open('data.b', 'rb')
records = np.fromfile(f, dtype='<i,<d,<d')
records
# array([(1, 2.3, 4.5), (6, 7.8, 9.0), (12, 13.4, 56.7)],
# dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<f8')])
record[0]
# (1, 2.3, 4.5)
record[1]
# (6, 7.8, 9.0)
# 当需要从媒体或者文件中读取二进制数据先去查找是否有相关的库

# 写一个二进制文件
import struct
import itertools

# 写数据
def write_polys(filename, polys):
    # Determine bounding box
    flattened = list(itertools.chain(*polys))
    min_x = min(x for x,y in flattened)
    max_x = max(x for x,y in flattened)
    min_y = min(y for x,y in flattened)
    max_y = max(y for x,y in flattened)
    with open(filename, 'wb') as f:
        f.write(struct.pack('<iddddi', 0x1234, min_x, min_y, max_x, max_y, len(polys)))
        for poly in polys:
            size = len(poly) * struct.calcsize('<dd')
            f.write(struct.pack('<i', size + 4))
            for pt in poly:
                f.write(struct.pack('<ddd', *pt))

# 读取数据
def read_polys(filename):
    with open(filename, 'rd') as f:
        # read the header
        header = f.read(40)
        file_code, min_x, min_y, max_x, max_y, num_polys=\
            struct.unpack('<iddddi', header)
        polys = []
        for n in range(num_polys):
            pbytes, = struct.unpack('<i', f.read(4))
            poly = []
            for m in range(pbytes // 16):
                pt = struct.unpack('<dd', f.read(16))
                poly.append(pt)
            polys.append(poly)
    return polys


import struct
class StructField:
    '''
    Descriptor representing a simple structure field
    '''
    def __init__(self, format, offset):
        self.format = format
        self.offset = offset
    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            r = struct.unpack_from(self.format, instance._buffer, self.offset)
            return r[0] if len(r) == 1 else r
        
class Structure:
    def __init__(self, bytedata):
        self._buffer = memoryview(bytedata)
# Structure类是一个基础类，接收字节数据并存储在内部的内存缓冲中，并被StructField描述器使用
# 可以定义一个高层次的结构对象来表示上表格信息所期望的文件格式
class PolyHeader(Structure):
    file_code = StructField('<i', 0)
    min_x = StructField('<d', 4)
    min_y = StructField('<d', 12)
    max_x = StructField('<d', 20)
    max_y = StructField('<d', 28)
    num_polys  = StructField('<d', 36)

# 如果你遇到了一些冗余的类定义，你可以考虑使用类装饰器或元类
class StructureMeta(type):
    '''
    Metaclass that automatically creates StructField descriptors
    '''
    def __init__(self, clsname, bases, clsdict):
        fields = getattr(self, '_fields_', [])
        byte_order = ''
        offset = 0
        for format, fieldname in fields:
            if format.startswith(('<', '>','!','@')):
                byte_order = format[0]
                format = format[1:]
            format = byte_order + format
            setattr(self, fieldname, StructField(format, offset))
            offset += struct.calcsize(format)
        setattr(self, 'struct_size', offset)

class Structure(metaclass=StructureMeta):
    def __init__(self, bytedata):
        self._buffer = bytedata
    
    @classmethod
    def from_file(cls, f):
        return cls(f.read(cls.struct_size))
class PolyHeader(Structure):
    _fields_ = [
        ('<i', 'file_code'),
        ('d', 'min_x'),
        ('d', 'min_y'),
        ('d', 'max_x'),
        ('d', 'max_y'),
        ('i', 'num_polys')
    ]

# 支持嵌套的字节结构
class NestedStruct:
    '''
    Descriptor representing a nested structure
    '''
    def __init__(self, name, struct_type, offset):
        self.name = name
        self.struct_type = struct_type
        self.offset = offset

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            data = instance._buffer[self.offset + self.offset + self.struct_type.struct_size] 
            result = self.struct_type(data)
            # Save resulting structure back on instance to avoid
            # further recomputation of this step
            setattr(instance, self.name, result)
            return result

class StructureMeta(type):
    '''
    Metaclass that automatically creates StructField descriptors
    '''
    def __init__(self, clsname, bases, clsdict):
        fields = getattr(self, clsname, bases, clsdict)
        byte_order = ''
        offset = 0
        for format, fieldname in fields:
            if isinstance(format, StructureMeta):
                setattr(self, fieldname, NestedStruct(fieldname, format, offset))
                offset += format.struct_size
            else:
                if format.startswith('<', '>', '!', '@'):
                    byte_order = format[0]
                    format = format[1:]
                format = byte_order + format
                setattr(self, fieldname, StructField(format, offset))
                offset += struct.calcsize(format)
        setattr(self, 'struct_size', offset)   

class Point(Structure):
    _fields_ = [
        ('<d', 'x'),
        ('d', 'y')
    ]

class PolyHeader(Structure):
    _fields_ = [
        ('<i', 'file_code'),
        (Point, 'min'), # nested struct
        (Point, 'max'), # nested struct
        ('i', 'num_polys')
    ]

# 写一个类来表示字节数据，写一个工具函数来通过多少方式解析内容
class SizedRecord:
    def __init__(self, bytedata):
        self.buffer = memoryview(bytedata)

    @classmethod
    def from_file(cls, f, size_fmt, includes_size=True):
        sz_nbytes = struct.calcsize(size_fmt)
        sz_bytes = f.read(sz_nbytes)
        sz, = struct.unpack(size_fmt, sz_bytes)
        buf = f.read(sz - includes_size * sz_nbytes)
        return cls(buf)
    
    def iter_as(self, code):
        if isinstance(code, str):
            s = struct.Struct(code)
            for off in range(0, len(self.buffer), s.size):
                yield s.unpack_from(self._buffer, off)
        elif isinstance(code, StructureMeta):
            size = code.struct_size
            for off in range(0, len(self._buffer), size):
                data = self._buffer[off:off+size]
                yield code(data)
# 从多边形文件中读取单独的多边形数据
f = open('polys.bin', 'rb')
phead = PolyHeader.from_file(f)
polydata = [SizedRecord.from_file(f, '<i') for n in range(phead.num_polys)]
# polydata
# [<__main__.SizedRecord object at 0x1006a4d50>,
# <__main__.SizedRecord object at 0x1006a4f50>,
# <__main__.SizedRecord object at 0x10070da90>]
# >>>
# iter_as 方法可以接受一个格式化编码或者Structure类作为输入，很灵活的去解析数据
for n, poly in enumerate(polydata):
    print('Polygon', n)
    for p in poly.iter_as('<dd'):
        print(p)

for n, poly in enumerate(polydata):
    print('Polygon', n)
    for p in poly.iter_as('<dd'):
        print(p.x, p.y)

# read_polys 的另一个修正版
class Point(Structure):
    _fields_ = [
        ('<d', 'x'),
        ('d', 'y'),
    ]

class PolyHeader(Structure):
    _fields_ = [
        ('<i', 'file_code'),
        (Point, 'min'),
        (Point, 'max'),
        ('i', 'num_polys'),
    ]
def read_polys(filename):
    polys = []
    with open(filename, 'rb') as f:
        phead = PolyHeader.from_file(f)
        for n in range(phead.num_polys):
            rec = SizedRecord.from_file(f, '<i')
            poly = [(p.x, p.y) for p in rec.iter_as(Point)]
            polys.append(poly)
    return polys
# memoryview的使用可以帮助我们避免内存的复制，当结构存在嵌套的时候，memoryview可以叠加一内存区域上定义的机构的不同部分
# 内存切片擦欧总不会拷贝，而是在已存在的内存上叠加，这样更高效
# 对任何涉及到统计、时间序列以及其他相关技术的数据分析问题，都可以使用pandas库

import pandas
rats = pandas.read_csv('rats.csv', skip_footer=1)
# investigate range of values for a certain field
rats['Current Activity'].unique()
# Filter the data
crew_dispatched = rats[rats['Current Activity'] == 'Dispatch Crew']
len(crew_dispatched)
# find 10 most rat-infested zip codes in chicago
crew_dispatched['ZIP Code'].value_counts()[:10]
# group by completion date
dates = crew_dispatched.groupby('Completion Date')
len(dates)
# determine counts on each day
date_counts = dates.size()
crew_dispatched[0:10]
# sort the counts
date_counts.sort()
date_counts[-10:]

