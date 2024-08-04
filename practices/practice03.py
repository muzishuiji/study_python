# string格式读取文件
with open('somefile.txt', 'rt') as f:
    data = f.read()

# 以行格式读取文件
with open('somefile.txt', 'rt') as f:
    for line in f:
        # process line


# 读写文件一般来讲是比较简单的，但是这几点也是需要注意的，首先，例子中的with语句给被使用到的文件创建了一个上下文环境
# 但with控制块结束时，文件会自动关闭，你也可以不使用with语句，但这是需要手动关闭文件
# wt模式默认会覆盖式写入，at模式是append写入
with open('somefile.txt', 'rt', encoding='latin-1') as f:
    #...
# 可使用newline=''来去除掉默认的换行符处理方式
with open('somefile.txt', 'rt', newline='') as f:
    #...

# 可以借助可选的errors参数来处理编码错误
# 使用unicode u+fffd来替换
f = open('sample.txt', 'rt', encoding='ascii', errors='replace')
f.read()
# 使用ignore来忽略
g = open('sample.txt', 'rt', encoding='ascii', errors='ignore')
g.read()

# 可以使用print打印文本内容，只支持打印文本模式打开的文件内容
# 可以使用sep和end关键字参数来指定打印的分隔符和行尾符
print('ACME', 50, 91.5, sep=',', end='!!\n')
# ACME,50,91.5!!
# 也可以将其转换成str，然后join
row = ('ACME', 50, 91.5)
print(','.join(str(x) for x in row))
# ACME,50,91.5
# 或者可以这样
print(*row, sep=',')
# 二进制格式读取和写入数据
with open('somefile.bin', 'rb') as f:
    data = f.read()

with open('somefile.bin', 'wb') as f:
    f.write(b'hello world')

# 读取二进制数据时，需要指明的是所有返回的数据都是字节字符串格式的，而不是文本字符串
# 写入的时候，必须保证参数是以字节形式对外暴露数据的对象（比如字节字符串，字节数组对象等）
# 对字节字符串迭代和索引返回的是字节的值而不是字节字符串
# 如果想要直接读取文本，需要进行解码和编码操作
b = b'Hello World'
b[0]
# 72
with open('somefile.bin', 'wb') as f:
    text = 'Hello World'
    f.write(text.encode('utf-8'))

# 二进制I/O还有一个鲜为人知的特性就是数组和c结构体类型能直接被写入，而不需要中间转换为自己对象
import array
nums = array.array('i', [1,2,3,4])
with open('data,bin', 'wb') as f:
    f.write(nums)

# 使用文件对象的readinto方法直接读取二进制数据到其底层的内存中去
import array
a = array.array('i', [0,0,0,0,0,0,0,0])
with open('data.bin', 'rb') as f:
    f.readinto() # 17
a
# array('i', [1, 2, 3, 4, 0, 0, 0, 0])
# 读取二进制数据可修改缓冲区
# 默认的wt会覆盖式写入，可通过xt来防止覆盖式写入
# python的旧版本或者是python实现的底层c函数库中没有这个模式
with open('test.txt', 'xt') as f:
    f.write('Hello4\n')

with open('test2.txt', 'xt') as f:
    f.write('Hello4\n')

# 创建类文件对象操作字符串数据
s = io.StringIO()
s.write('hello world\n')
# 12
print('this is a test', file=s)
# 15
# 获取当前写入的所有疏浚
s.getValue()
# 包装一个文件接口
s = io.StringIO('hello\nworld\n')
s.read(4)
# 'Hell
s.read()
# 'o\nworld\n'

# BytesIO 可以用来操作二进制数据
s = io.BytesIO()
s.write(b'binary data')
s.getValue()
# b'binary data'
# 当你想摸你一个普通的文件的时候StringIO和BytesIO类都是很有用的
# 比如，在单元测试中，你可以使用StringIO来创建一个包含测试数据的类文件对象，这个对象可以被传给某个参数为普通文件对象的函数
# StringIO和BytesIO实例并没有正确的整数类型的文件描述符，因此，它们不能使用在需要使用真实的系统文件如文件、管道或者套接字的程序中。

# 可以使用gzip，和bz2来读取和写入压缩文件
import gzip
with gzip.open('somefile.gz', 'rt') as f:
    text = f.read()

import bz2
with bz2.open('somefile.bz2', 'rt') as f:
    text = f.read()

# 写入压缩数据
with gzip.open('somefile.gz', 'wt') as f:
    f.write(text)

# compresslevel 可以用来制定压缩击毙，最高等级是9，默认的压缩登记
# 压缩等级越低性能越好，但数据压缩的程度也越低
import bz2
with bz2.open('somefile.bz2', 'wt', compresslevel=5) as f:
    f.write(text)

# gzip.open 和 bz2.open还有一个很少被知道的属性，可以坐拥在一个已存在并以二进制模式打开的文件上
import gzip
# 已存在并以二进制模式打开
f = open('somefile.gz', 'rb')
with gzip.open(f, 'rt') as g:
    text = g.read()
# 这个就允许gzip和bz2模块可以工作在许多文件对象上，比如套接字、管道和内存中文件等。

# 如果想在一个固定长度记录或数据块的集合上迭代，而不是一行一行的迭代
from functools import partial
RECORD_SIZE = 32
with open('somefile.data', 'rd') as f:
    # records 对象是一个可迭代对象，它会不断产生固定大小的数据块，直到文件末尾
    records = iter(partial(f.read, RECORD_SIZE, 'b'))
    for r in records:
        print(r)
# iter函数有一个特点，如果你给它传递一个可调用对象和一个标记值，它会创建一个迭代器，这个迭代器会一直调用传入的可调用对象直到它返回标记值位置。
# 读取数据到一个可变数组中，使用文件对象的readinto方法
import os.path

def read_into_buffer(filename):
    buf = bytearray(os.path.getsize(filename))
    with open(filename, 'rb') as f:
        # readinto方法能被用来为预先分配内存的数组填充数据，
        # 和 read方法不同的是，readinto方法填充已存在的缓冲区而不是为新对象重新分配内存再返回它们
        # 因此，可以用它避免大量的内存分配操作
        f.readinto(buf)
    return buf

with open('sample.bin', 'wb') as f:
    f.write(b'Hello World')
buf = read_into_buffer('sample.bin')
print(buf)
# bytearray(b'Hello World')
buf[0:5] = b'Hello'
print(buf)
# 11
with open('newsample.bin', 'wb') as f:
    f.write(buf)

# 读取一个相同大小的记录组成的二进制文件
record_size = 32
buf = bytearray(record_size)
with open('somefile', 'rb') as f:
    while True:
        n = f.readinto(buf)
        if n < record_size:
            break

m1 = memoryview(buf)
m2 = m1[-5:]
m2[:] = b'WORLD'
# memoryview通过领复制的方式对已存在的缓冲区执行切片操作，甚至还能修改他的内容

print(buf)
# bytearray(b'Hello WORLD')

# 使用mmap模块来内存映射文件
import os
import mmap

def memory_map(filename, access=mmap.ACCESS_WRITE):
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)

# 创建一个文件，将其内容扩充到指定大小
size = 1000000
with open('data', 'wb') as f:
    f.seek(size-1)
    f.write(b'\x00')

# 用memory_map来内存映射文件内容
m = memory_map('data')
len(m)
# 1000000
m[0:10]
# b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
m[0]
# 0
# 做切片操作
m[0:10] = b'Hello World'
m.close()

# 
with open('data', 'rb') as f:
    print(f.read(11))
# b'Hello World'
# mmap()返回的mmap对象同样可以作为一个上下文管理器来使用，这时候底层的文件会被自动岩壁
with memory_map('data') as m:
    print(len(m))
    print(m[0:10]) 
# 1000000
# b'Hello World'
m.closed # True
# memeory_map 打开的文件支持读写操作，会同步修改内容到原来的文件中
# 只读的访问模式
m = memory_map(filename, mmap.ACCESS_READ)
# 只修改本地文件，不修改原文件
m = memory_map(filename, mmap.ACCESS_COPY)
# mmap 所暴露的内存看上去是一个二进制数组对象，但是，可以使用一个内存试图来解析其中的数据
m = memory_map('data')
v = memoryview(m).cast('I')
v[0] = 7
m[0:4]
# b'\x07\x00\x00\x00'
m[0:4] = b'\x07\x01\x00\x00'
v[0] # 263

# 如果多个python解释器内存映射同一个文件，得到的mmap对象能够用来再解释其中直接交换数据。也就是说，所有的解释器都能同时读写数据
# 并且其中一个解释器所做的修改会自动呈现在其他解释器中。很明显，这里需要考虑同步的问题，
# 但是这种方法有时候可以用来在管道或套接字间传递数据。
# 使用路径名来获取文件名、目录名、绝对路径
import os
path = '/Users/beazley/Data/data.csv'
os.path.basename(path)
# data.csv

os.path.join('tmp', 'data', os.path.basename(path))
# 'tmp/beazley/Data

path = '~/Data/data.csv'
os.path.expanduser(path)
# 'data.csv'

os.path.dirname(path)
# '/Users/beazley/Data'

os.path.join('tmp', 'data', os.path.basename(path))
# 'tmp/data/data.csv'

path = '~/Data/data.csv'
os.path.expanduser(path)
# '/Users/beazley/Data/data.csv'

os.path.splitext(path)
# ('~/Data/data', '.csv')

# 对于任何的文件名操作，你都应该使用os.path模块，而不是自己封装一些字符串操作方法来造轮子
# 测试文件是否存在
import os
os.path.exists('/etc/passwd')
# True
os.path.exists('/tmp/spam')
# False

# Is a regular file.
os.path.isfile('/etc/passwd')
# 是否是文件夹
os.path.isdir('/etc/passwd')

# 是否是可执行文件
os.path.islink('/etc/passwd')
# 是否是文件索引？
os.path.realpath('/etc/passwd')
# '/usr/local/bin/python3.3'
# 获取文件大小
os.path.getsize('/etc/passwd')
# 修改日期
os.path.getmtime('/etc/passwd')
# 1272478234.0
import time
time.ctime(os.path.getmtime('/etc/passwd'))
# 'Wed Apr 28 13:10:34 2010'
# 使用os.path来操作文件的时候需要注意文件权限的问题

# listdir获取某个目录的文件列表
import os.path
names = [name for name in os.listdir('somedir') if os.path.isfile(os.path.join('somedir', name))]

# startswith, endswith来过滤
pyfiles = [name for name in os.listdir('somedir') if name.endswith('.py')]

# glob,fnmatch
import glob
# 过滤出文件名匹配正则的文件
pyfiles = glob.glob('somedir/*.py')

from fnmatch import fnmatch
pyfiles = [name for name in os.listdir('somedir') if fnmatch(name, '*.py')]

import os
import os.path
import glob

pyfiles = glob.glob('*.py')

# Get file sizes and modification dates
name_sz_date = [(name, os.path.getsize(name), os.path.getmtime(name))
                for name in pyfiles]
for name, size, mtime in name_sz_date:
    print(name, size, mtime)

# Alternative: Get file metadata
file_metadata = [(name, os.stat(name).st_size, os.stat(name).st_atime) for name in pyfiles]
for name, size,mtime  in file_metadata:
    print(name, size, mtime)

# 获取文件名的编码
sys.getfilesystemencoding()
# 'utf8'

# 如果因为某种原因你想忽略这种编码，则使用一个原始字节字符串来指定一个文件名即可
with open('jalape\xf1o.txt', 'w') as f:
    f.write('Spicy!')
# 6

import os
os.listdir('.')
# ['jalapeño.txt']

os.listdir(b'.')
# [b'jalapen\xcc\x83o.txt']
# open file with raw filname
with open(b'jalapen\xcc\x83o.txt') as f:
    print(f.read())

# 读取目录并通过原始未解码的方式处理文件名可以有效避免文件名中不符合默认编码的文件导致程序中断。
# 当读取文件名包含到不合法的字符时，python的解决方案是从文件名中获取未解码的字节值比如、xhh并将它映射成unicode字符
# \udchh表示的所谓的“代理编码
import os
files = os.listdir('.')
files
# ['spam.py', 'bäd.txt', 'foo.txt']
# ['spam.py', 'b\udce4d.txt', 'foo.txt']

# 当打印位置的文件名时，使用下面的方法可以避免这样的错误
def bad_filename(filename):
    return repr(filename)[1:-1]
# 直接打印包含不合法字符的文件名可能会导致程序崩溃
try:
    print(filename)
except UnicodeEncodeError:
    print(bad_filename(filename))

# 在bad_filename中通过某种方式对文件名重新编码
def bad_filename(filename):
    temp = filename.encode(sys.getfilesystemencoding(), errors='surrogateescape')
    return temp.decode('latin-1')
# surrogateescape 是puthon在绝大部分面向OS的api所使用的错误处理器
# 它能以一种优雅的方式处理由操作系统提供的数据的编码问题
# 在解码出错时会将出错字节存储到一个很少被使用到的Unicode编码范围内。
# 在编码时将那些隐藏值又还原回原先解码失败的字节序列。
# 它不仅对于OS API非常有用，也能很容易的处理其他情况下的编码错误。
# 当你在编写依赖文件名和文件系统的关键任务程序时，就必须得考虑遇到文件名中包含不合法字符时如何处理

# 改变已经打开的一个二进制文件的编码
import urllib.request
import io

u = urllib.request.urlopen('http://www.python.org')
f = io.TextIOWrapper(u, encoding='utf-8')
text = f.read()

# 改变一个已经打开的文本模式的文件的编码
import sys
sys.stdout.encoding
# 'UTF-8'
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='latin-1')
# 'latin-1'
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='ascii', errors='xmlcharrefreplace')
print('Jalape\u00f1o')
# 改变编码方式从而改变行的处理方式和错误处理的方式

# 文本文件是通过一个拥有缓冲的二进制模式文件上增加一个unicode编码/解码来创建，buffer属性只想对应的底层文件，如果直接访问它的话就会绕过文本编码/解码层
# sys.stdout总是以文本模式打开的，需要打印二进制数据到标准输出的脚本的话，可以用下面的方式
import sys
sys.stdout.buffer.write(b'hello\n')

# open方法可以使用open来将文件描述符包装成一个python文件对象
import os
fd = os.open('somefile.txt', os.O_WRONLY | os.O_CREAT)

# turn into a proper file
f = open(fd, 'wt')
f.write('hello world\n')
f.close()

# 将文件描述符包装成文件对象
from socket import socket, AF_INET, SOCK_STREAM

def echo_client(client_sock, addr):
    print('got connection from', addr)
    # make text-mode file wrappers for socket reading/writing
    client_in = open(client_sock.lineno(), 'rt', encoding='latin-1', closefd=False)
    client_out = open(client_sock.lineno(), 'wt', encoding='latin-1', closefd=False)

    # echo lines  back to the client using file I/O    
    for line in client_in:
        client_out.write(line) 
        client_out.flush()
    client_sock.close()


def echo_server(address):
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(address)
    sock.listen(1)
    while True:
        client, addr = sock.accept()
        echo_client(client, addr)

# 使用TemporaryFile(), NamedTemporaryFile(), TemporaryDirectory函数来处理临时文件目录
# 它们会自动处理所有的创建和清理步骤
        
# 与串行端口通信
import serial
ser = serial.Serial('/dev/tty.su', baudrate=9600,bytesize=8,parity='N', stopbits=1)

ser.write(b'G1 X50 Y50\r\n')
resp = ser.readline()
# 串口通信的高级特性：超时、控制流，缓冲区刷新，握手协议等
# 涉及到串口通信的IO都是二进制模式的，确保使用的是字节而不是文本，或者执行文本的编码/解码擦欧总
# 当你需要创建二进制编码的指令或数据包的时候，struct模块也是非常有用的

# pickle.load在加载时会有一个副作用就是它会自动加载相应模块并构造实例对象
# 可以利用pickle的工作原理，创建一个恶意的数据导致python执行随意指定的系统命令，要保证pickle只在相互之间可以认证对方的解析器的内部使用

# 对于一些不能被序列化的类型的对象，可以通过自定义提供__getstate__ 和__setstate__来序列化和反序列化
# 长期存储的数据不应该选择pickle
# pickle 用于序列化python对象
import pickle
data = '....'
f = open('somefile', 'wb')
pickle.dump(data, f)
# 转储为一个字符串
s = pickle.dumps(data)
# 从字节流中恢复一个对象
f = open('somefile', 'rb')
data = pickle.load(f)
# 从字节流中恢复一个字符串
data = pickle.loads(s)
# pickle 可以按顺序序列化和反序列化多个对象