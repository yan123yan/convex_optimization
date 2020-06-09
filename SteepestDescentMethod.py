import numpy as np
import matplotlib.pyplot as plt

#无约束条件下的最速下降法（Rosenbrock函数）

# 定义x, y, 数据数量为256
from matplotlib import ticker

x_value = np.linspace(-1,1.1,256)
y_value = np.linspace(-0.1,1.1,256)

def f(x,y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2

def df(x,y):
    descent = np.array([[400 * x * x * x + 2 * x - 2 - 400 * x * y],
                        [200 * y - 200 * x * x]])
    return descent

# 生成网格数据
X, Y = np.meshgrid(x_value, y_value)
#显示等高线
plt.figure()
# 填充等高线的颜色, 8是等高线分为几部分
plt.contourf(X, Y, f(X, Y), 5, alpha=0)
# 绘制等高线
C = plt.contour(X, Y, f(X, Y), 8, locator=ticker.LogLocator(), colors='black', linewidth=0.01)
# 绘制等高线数据
plt.clabel(C, inline=True, fontsize=10)


alpha = 0.09
beta = 0.8
x = np.array([[-0.2],[0.4]]) #随机选取一个起始点
eta = 0.01 #学习率

#print(df(x))

xv = [x[0,0]]
yv = [x[1,0]]
plt.plot(x[0, 0], x[1, 0], marker='o')
#plt.show()

t_list = []
dx = np.zeros((2,1))

while (abs(df(x[0,0],x[1,0])[0,0]) > eta and abs(df(x[0,0],x[1,0])[1,0]) > eta):
    #确定下降方向
    #二范数
    # if df(x[0,0],x[1,0])[0] > 0:
    #     dx[0,0] = -np.linalg.norm(df(x[0,0],x[1,0])[0],ord=2)
    # else:
    #     dx[0,0] = np.linalg.norm(df(x[0,0], x[1,0])[0], ord=2)
    #
    # if df(x[0,0],x[1,0])[1] > 0:
    #     dx[1,0] = -np.linalg.norm(df(x[0,0],x[1,0])[1], ord=2)
    # else:
    #     dx[1,0] = np.linalg.norm(df(x[0,0], x[1,0])[1], ord=2)
    #
    # print(dx)

    #非规范化的一范数
    # if df(x[0, 0], x[1, 0])[0] > 0:
    #     dx[0, 0] = -np.linalg.norm(df(x[0, 0], x[1, 0])[0], ord=1)
    # else:
    #     dx[0, 0] = np.linalg.norm(df(x[0, 0], x[1, 0])[0], ord=1)
    # if df(x[0, 0], x[1, 0])[1] > 0:
    #     dx[1, 0] = -np.linalg.norm(df(x[0, 0], x[1, 0])[1], ord=1)
    # else:
    #     dx[1, 0] = np.linalg.norm(df(x[0, 0], x[1, 0])[1], ord=1)

    #规范化的一范数
    if df(x[0, 0], x[1, 0])[0] > 0:
        dx[0,0] = -1
    elif df(x[0, 0], x[1, 0])[0] == 0:
        dx[0, 0] = 0
    else:
        dx[0,0] = 1

    if df(x[0, 0], x[1, 0])[1] > 0:
        dx[1,0] = -1
    elif df(x[0, 0], x[1, 0])[1] == 0:
        dx[1, 0] = 0
    else:
        dx[1,0] = 1

    print(dx)

    #回溯直线搜索
    t = 1
    t_list.append(t)

    f_x_t_delta_x = f(x[0,0]+t*dx[0,0],x[1,0]+t*dx[1,0])
    print("f_x_t_delta_x",f_x_t_delta_x)
    fx = f(x[0,0],x[1,0])
    print("fx", fx)
    #print(np.dot(df(x[0,0],x[1,0]).T, dx))
    f_x_a_t_grad_delta_x = fx + alpha * t * np.dot(df(x[0,0],x[1,0]).T, dx)
    print("f_x_a_t_grad_delta_x", f_x_a_t_grad_delta_x)

    while (f(x[0,0]+t*dx[0,0],x[1,0]+t*dx[1,0]) > f(x[0,0],x[1,0]) + alpha * t * np.dot(df(x[0,0],x[1,0]).T, dx)):
        t = beta * t
        t_list.append(t)

    # 修改x
    x = x + t * dx
    xv.append(x[0, 0])
    yv.append(x[1, 0])

    #descent = 2 * x

plt.plot(xv,yv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Steepest Descent Method for Rosenbrock Function (L2)')
plt.legend()
plt.show()