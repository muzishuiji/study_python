# 约束类的属性类型
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('name must be a string')
        self._name = value
    
    @property
    def age(self):
        return self._age 
    
    @age.setter
    def age(self, value):
        if not isinstance(value, int):
            raise TypeError('age must be an int')
        self._age = value

# 为了避免重复代码，可以创建一个函数定义属性并返回（set里做类型校验）
# 生成属性并返回这个属性对象
def typed_property(name, expected_type):
    storage_name = '_' + name

    @property
    def prop(self):
        return getattr(self, storage_name)
    
    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a {}'.format(name, expected_type))
        setattr(self, storage_name, value)

    return prop

# example use
class Person:
    name = typed_property('name', str)
    age = typed_property('age', int)

    def __init__(self, name, age):
        self.name = name
        self.age = age

# 使用functools.partial来稍微改变下
from functools import partial
String = partial(typed_property, expected_type=str)
Integer = partial(typed_property, expected_type=int)
class Person:
    name = String('name')
    age = Integer('age')

    def __init__(self, name, age):
        self.name = name
        self.age = age
    
# 实现一个上下文管理器的最简单的方法就是使用contextlib模块中的@contextmanager装饰器
import time
from contextlib import contextmanager

@contextmanager
def timethis(label):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print('{}: {}'.format(label, end - start))
# example use
with timethis('counting'):
    n = 1000000
    while n > 0:
        n -= 1

# 高级一点的上下文管理器，实现了列表上的某种事务
@contextmanager
def list_transaction(orig_list):
    working = list(orig_list)
    yield working
    orig_list[:] = working

# 上述代码是任何对列表的修改只有当前所有代码运行完成且不出现异常的情况下才会生效
items = [1,2,3]
with list_transaction(items) as working:
    working.append(4) 
    working.append(5) 

items # [1, 2, 3, 4, 5]

with list_transaction(items) as working:
    working.append(6) 
    working.append(7) 
    raise RuntimeError('oops')

# Traceback (most recent call last):
#     File "<stdin>", line 4, in <module>
# RuntimeError: oops
items # [1, 2, 3, 4, 5]

# 自定义一个上下文管理器，需要实现enter方法和exit方法
import time
class timethis:
    def __init__(self, label):
        self.label = label
    
    def __enter__(self):
        self.start = time.time()
    
    def __exit__(self, exc_ty, exc_val, exc_tb):
        end = time.time()
        print('{}: {}'.format(self.label, end - self.start))

# @contextmanager只能为函数提供上下文管理的能力，有一些对象（文件、网络连接或锁），需要支持with语句，则需要单独实现enter和exit方法

# 全局命名空间内执行一个代码片段
a = 13
exec('b = a + 1')
print(b) # 14

def test():
    a = 13
    exec('b = a + 1')
    print(b)
test()
# Traceback (most recent call last):
#     File "<stdin>", line 1, in <module>
#     File "<stdin>", line 4, in test
# NameError: global name 'b' is not defined
# 上述代码exec的内容执行后所有结果都不可见，如果需要保留结果，则需要用到一个局部变量字典，从局部变量字典中获取修改过的变量值
def test():
    a = 13
    loc = locals()
    exec('b = a + 1')
    b = loc['b']
    print(b)
test() # 14
# 实际上对exec的正确使用是比较难的，大多数情况下你要考虑使用exec的时候，还有另外更好的解决方案（如装饰器，闭包，元类等）
# exec处理局部变量实际上会先做一个拷贝，所以对局部变量的修改不会影响局部变量
def test1():
    x = 0
    exec('x += 1')
    print(x)
test1() # 0
# 当调用locals获取局部变量时，获得的是传递给exec的局部变量的一个拷贝
# 通过在代码执行后审查这个字典的值，就能获取修改后的值了
def test2():
    x = 0
    loc = locals()
    print('before:', loc)
    exec('x += 1')
    print('after:', loc)
    print('x =', x)

test2()
# before: {'x': 0}
# after: {'loc': {...}, 'x': 1} loc被赋值给x的拷贝
# x = 0

# locals() 会获取局部变量中的值并覆盖字典中相应的变量
def test3():
    x = 0
    loc = locals()
    print(loc)
    exec('x += 1')
    print(loc)
    # locals()的值被覆盖了
    locals()
    print(loc)
test3()
# {'x':0}
# {'loc': {...}, 'x': 1}
# {'loc': {...}, 'x': 0}

# 作为locals()的一个替代方案，你可以使用自己的字典，并将它传递个exec()
def test4():
    a = 13
    loc = {'a': a}
    glb = {}
    exec('b = a + 1', glb, loc)
    b = loc['b']
    print(b)
test4() # 14

# ast模块用来将python源码编译成一个可被分析的抽象语法树（ast）
import ast
ex = ast.parse('2 + 3*4 + x', mode='eval')
ast.dump(ex)
# "Expression(body=BinOp(left=BinOp(left=Num(n=2), op=Add(),
# right=BinOp(left=Num(n=3), op=Mult(), right=Num(n=4))), op=Add(),
# right=Name(id='x', ctx=Load())))"
top = ast.parse('for i in range(10): print(i)', mode='exec')
ast.dump(top)
# "Module(body=[For(target=Name(id='i', ctx=Store()),
# iter=Call(func=Name(id='range', ctx=Load()), args=[Num(n=10)],
# keywords=[], starargs=None, kwargs=None),
# body=[Expr(value=Call(func=Name(id='print', ctx=Load()),
# args=[Name(id='i', ctx=Load())], keywords=[], starargs=None,
# kwargs=None))], orelse=[])])"

# 实现一个访问者类，记录哪些名字被加载、存储和删除
import ast
class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.loaded = set()
        self.stored = set()
        self.deleted = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.stored.add(node.id)
        elif isinstance(node.ctx, ast.Del):
            self.deleted.add(node.id)

# sample usage
if __name__ == '__main__':
    # some python code
    code = '''
    for i in range(10):
        print(i)
    del i
    '''

    # parse into an ast
    top = ast.parse(code, mode='exec')
    # feed the ast to analyze name usage
    c = CodeAnalyzer()
    c.visit(top)
    print('Loaded:', c.loaded)
    print('Stored:', c.stored)
    print('Deleted:', c.deleted)

# 运行后得到如下输出
Loaded: {'i', 'range', 'print'}
Stored: {'i'}
Deleted: {'i'}

# ast可以通过compile()函数来编译并执行
exec(compile(top, '<stdin>', 'exec'))
# 0 1 2 3 4 5 6 7 8 9

# 通过重新解析函数体源码、重写ast并重新创建函数代码对象来将全局变量将为函数作用范围
import ast
import inspect
# node visitor that lowers globally accessed names into the
# function body as local variables
class NameLower(ast.NodeVisitor):
    def __init__(self, lowered_names):
        self.lowered_names = lowered_names
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        # compile some assignments to lower the constants
        code = '__globals = globals()\n'
        code += '\n'.join("{0} = __globals['{0}']".format(name) for name in self.lowered_names)
        code_ast = ast.parse(code, mode='exec')
        # inject new statements into the function body
        # 将转换后的body替换node的body
        node.body[:0] = code_ast.body
        # save the function object
        self.func = node

# decorator that turns global names into locals
def lower_names(*namelist):
    def lower(func):
        srclines = inspect.getsource(func).splitlines()
        # skip source lines prior to the @lower_names decorator
        # 跳过@lower_names装饰器之前的源代码行
        for n, line in enumerate(srclines):
            if '@lower_names' in line:
                break
        src = '\n'.join(srclines[n+1:])
        # hack to deal with indented code
        # 处理代码缩进的技巧
        if src.startswith(' ', '\t'):
            src = 'if 1:\n' + src
        top = ast.parse(src, mode='exec')
        # transform the ast
        # 转换成ast
        cl = NameLower(namelist)
        cl.visit(top)
        # execute the modified ast
        # 执行修改后的ast
        temp = {}
        exec(compile(top, '', 'exec'), temp, temp)
        # pull out the modified code object
        # 取出修改后的代码对象
        func.__code__ = temp[func.__name__].__code__
        return func
    return lower
INCR = 1
@lower_names('INCR')
def countdown(n):
    while n > 0:
        n -= INCR

# 装饰器会讲countdown()函数重写为类似下面这样子
def countdown(n):
    __globals = globals()
    INCR = __globals['INCR']
    while n > 0:
        n -= INCR
# 使用ast是一个更加高级点的技术，可能对一些场景会简单些

# dis模块可以被用来输出任何python函数的反编译结果
def countdown(n):
    while n > 0:
        print('T-minus', n)
        n -= 1
    print('Blastoff')

import dis
dis.dis(countdown)
#   2           0 SETUP_LOOP              30 (to 32)
#         >>    2 LOAD_FAST                0 (n)
#               4 LOAD_CONST               1 (0)
#               6 COMPARE_OP               4 (>)
#               8 POP_JUMP_IF_FALSE       30

#   3          10 LOAD_GLOBAL              0 (print)
#              12 LOAD_CONST               2 ('T-minus')
#              14 LOAD_FAST                0 (n)
#              16 CALL_FUNCTION            2
#              18 POP_TOP

#   4          20 LOAD_FAST                0 (n)
#              22 LOAD_CONST               3 (1)
#              24 INPLACE_SUBTRACT
#              26 STORE_FAST               0 (n)
#              28 JUMP_ABSOLUTE            2
#         >>   30 POP_BLOCK

#   5     >>   32 LOAD_GLOBAL              0 (print)
#              34 LOAD_CONST               4 ('Blastoff!')
#              36 CALL_FUNCTION            1
#              38 POP_TOP
#              40 LOAD_CONST               0 (None)
#              42 RETURN_VALUE
# 当你想知道你的程序的底层运行机制的时候，dis模块是很有用的，比如你试着理解行呢特征
countdown.__code__.co_code
# b"x'\x00|\x00\x00d\x01\x00k\x04\x00r)\x00t\x00\x00d\x02\x00|\x00\x00\x83
# \x02\x00\x01|\x00\x00d\x03\x008}\x00\x00q\x03\x00Wt\x00\x00d\x04\x00\x83
# \x01\x00\x01d\x00\x00S"
# 如果你想自己解释这段代码，需要使用一些opcode模块中定义的常量
c = countdown.__code__.co_code
import opcode
opcode.opname[c[0]]
# 'SETUP_LOOP'
opcode.opname[c[2]]
# 'LOAD_FAST'
# 将原始字节码序列转换成opcodes和参数
import opcode
def generate_opcodes(codebytes):
    expended_arg = 0
    i = 0
    while i < n:
        op = codebytes[i]
        i += 1
        if op >= opcode.HAVE_ARGUMENT:
            oparg = codebytes[i] + codebytes[i+1]*256 + expended_arg
            expended_arg = 0
            i += 2
            if op == opcode.EXTENDED_ARG:
                expended_arg = oparg * 65536
                continue
            else:
                oparg = None
            yield (op, oparg)
for op, oparg in generate_opcodes(countdown.__code__.co_code):
    print(op, opcode.opname[op], oparg)

# 用generate_opcodes来替换任何想要替换的原始字节码
def add(x, y):
    return x + y
c = add.__code__
c
# <code object add at 0x1007beed0, file "<stdin>", line 1>
c.co_code
# b'|\x00\x00|\x01\x00\x17S'
# Make a completely new code object with bogus byte code
import types
newbytecode = b'xxxxxx'
nc = type.CodeType(c.co_argcount, c.co_kwonlyargcount,
                   c.co_nlocals, c.co_stacksize,c.co_flags,newbytecode,c.co_consts,c.co_names, c.co_varnames,c.co_filename,c,co_name,
                   c.co_firstlineno, c.co_lnotab)
nc
# <code object add at 0x10069fe40, file "<stdin>", line 1>
add.__code__ = nc
abb(2, 3) # Segmentation fault

