# 定义一个模块的层级包：在文件系统上组织你的代码，并确保每个目录都定义了一个__init__.py文件
# import graphics.primitive.line
# from graphics.primitive import line
# import graphics.formats.jpg as jpg

# 使用form module import * 语句时，可以通过使用__all__来定义导出内容，默认导入所有不以下划线开头的

# somemodule.py
def spam():
    pass

def grok():
    pass

blah = 42
# only export 'spam' and 'grok'
__all__ = ['spam', 'grok']

# 使用相对路径导入包中子模块
# mypackage/A/spam.py
from . import grok
# 绝对路径导入的问题是会讲顶层包名硬编码到你的源码中，如果有人想安装两个不同版本的软件包，只通过名称区分它们
# 如果使用相对路径，则没问题，使用绝对路径名很可能会有问题

import mymodule
a = mymodule.A()
a.spam()
# A.spam
b = mymodule.B()
b.bar()
# B.bar

# 延迟加载的代码示例
def A():
    from .a import A
    return A()

def B():
    from .b import B
    return B()

import mymodule
a = mymodule.A()
a.spam()
# A.spam
# # 延迟加载的主要缺点是继承和类型检查可能会中断，你可能需要额外的类型判断
# if isinstance(x, mymodule.A): # error
# if isinstance(x, mymodule.a.A): # ok


# 为不同的目录的代码安装到一个共同的命名空间，对于的框架，这可能是有用的，因为它允许一个框架的部分被单独的安装系在
# 它也使人们能够轻松的为这样的框架编写第三方附加组件和其他扩展

# 假设python代码的两个不同的目录如下：
# foo-package/
#     spam/
#         blah.py
# bar-package/
#     spam/
#         grok.py

# 这两个目录有共同的命名空间spam，加入模块路径并导入后，可以将不同的包目录合并到一起，
# 你可以导入spam.blah和spam.grok，并且它们能够工作
# import sys
# sys.path.extend(['foo-package', 'bar-package'])
# import spam.blah
# import spam.grok

# 包命名空间的关键是确保顶级目录中没有init.py文件来作为共同的命名空间，缺失init.py文件使得导入包的时候会发生有趣的事情
# 解释器创建了一个由所有包含匹配包名的目录组成的列表，特殊的包命名空间模块被创建，只读的目录列表副本被存储在其path变量中
import spam
spam.__path__
# _NamespacePath(['foo-package/spam', 'bar-package/spam'])

# 你可以像这样定义代码目录将你的包无缝的加入到别的spam包目录中
# my-package/
#     spam/
#         custom.py
# import spam.custom
# import spam.grok
# import spam.blah
# 一个包是否作为命名空间的两个方法：1. 检测file属性是否存在 2. 字符表现形式中的namespance
spam.__file__
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
# AttributeError: 'module' object has no attribute '__file__'
spam
# <module 'spam' (namespace)>

# 可以使用imp.reload()来重新加载先前加载的模块，这在开发和提示过程常常很有用，但在生产环境中的代码使用会很不安全
# reload不会更新像from module import name这样使用import语句导入的定义
# reload更适合在调试环境使用
# spam.py
def bar():
    print('bar')

def grok():
    print('grok')

# 交互式对话
# import spam
# from spam import grok
# spam.bar()
# bar
grok()
# grok

# 修改grok的源码
def grok():
    print('new grok')

import imp
imp.reload(spam)
# <module 'spam' from './span.py'>
spam.bar()
# bar
grok()
# grok
spam.grok()
# new grok

# 应用包含多个文件，可以提供一些简单的方法运行这个程序
# 创建目录
# myapplication/
#     spam.py
#     bar.py
#     grok.py
#     __main__.py

# main.py存在，可以简单的在顶级目录运行python解释器，解释器将执行main.py作为主程序
# bash % python3 myapplication
# 代码打包成zip文件也同样适用
# bash % ls
# spam.py bar.py grok.py __main__.py
# bash % zip -r myapp.zip *.py
# bash % python3 myapp.zip
# ... output from __main__.py ...

# 由于目录和zip文件与正常文件有一点不同，你可能哈需要增加一个shell脚本，使执行更加容易
# 可创建一个顶级脚本
# !/usr/bin/env python3 /usr/local/bin/myapp.zip

# 如果你的包中包含代码需要去读取的数据文件，你需要尽可能用最便捷的方式来做这件事
# mypackage/
#     __init__.py
#     somedata.dat
#     spam.py

# 读取数据内容
import pkgutil
data = pkgutil.get_data(__package__, 'somedata.dat')
# data: 包含该文件的原始内容的字节字符串

# 创建一个.pth文件放在python的site-packages目录，当解释器启动时，。pth文件里列举出来的存在于文件系统的目录将被添加到sys.path
# 安装一个pth文件可能需要管理员权限，如果被添加到系统级的python解释器
# 写一个代码手动调节sys.path的值
import sys
sys.path.insert(0, '/some/dir')
sys.path.insert(0, '/other/dir')
# 上述代码的问题是代码移动到新的位置，会导致维护问题
# 可以使用模块级的变量来精心构造一个适当的绝对路径，比如__file__
import sys
from os.path import abspath,join,dirname
sys.path.insert(0, join(abspath(dirname(__file__)), 'src'))
# 你可以把你的代码放在一系列不同的目录，只要哪些目录包含在.pth文件里

# 使用importlib.import_module 来导入模块或者包的一部分
import importlib
math = importlib.import_module('math')
math.sin(2)
# 0.9092974268256817
mod = importlib.import_module('urllib.request')
u = mod.urlopen('http://www.python.org')
# import module 只是简单执行和import相同的步骤，但是返回生成的模块对象，你只需要存储在一个变量，然后像正常的模块一样使用
# import_module()也可使用相对导入，需要一个额外的参数
import importlib
# Same as 'from . import b'
b = importlib.import_module('.b', __package__)
# import_module 主要用于修改或覆盖模块的代码时候

# 创建一个从服务器加载源代码的工具函数
import imp
import urllib.request
import sys

# 下载源代码，使用compile将其编译到一个代码对象中，然后在一个新创建的模块对象的字典中来执行它
def load_module(url):
    u = urllib.request.urlopen(url)
    source = u.read().decode('utf-8')
    mod = sys.modules.setdefault(url, imp.new_module(url))
    # 将文件源代码编译到一个代码对象中，然后在一个新创建的模块字典中执行它
    code = compile(source, url, 'exec')
    mod.__file__ = url
    mod.__package__ = ''
    exec(code, mod.__dict__)
    return mod

fib = load_module('http://localhost:15000/fib.py')
# I'm fib


from urllib.request import urlopen
u = urlopen('http://localhost:15000/fib.py')
data = u.read().decode('utf-8')
print(data)