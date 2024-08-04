# import numpy as np
# import matplotlib.pyplot as plt

# # 生成数据,以0.1为单位，生成0-6的数据
# x = np.arange(0, 6, 0.1)
# y1 = np.sin(x)
# y2 = np.cos(x)


# # 绘制图形
# plt.plot(x, y1, label="sin")
# # 用虚线绘制
# plt.plot(x, y2, linestyle = "--",label="cos")
# # 加x轴标签，y轴标签
# plt.xlabel("x")
# plt.ylabel("y")
# # 加标题
# plt.title('sin & cos')
# # 加图例
# plt.legend()
# plt.show()


# matplotlib 读区图像
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
try:
    img = mpimg.imread('test.png')
    plt.imshow(img)
    plt.show()
except FileNotFoundError:
    print("文件不存在")
except Exception as e:
    print(f"An error occurred: {e}")

# python 可以将一系列处理集成为函数或类模块