import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

#障碍方法（Rosenbrock函数）

# 定义x, y, 数据数量为256
x_value = np.linspace(-1,1.1,256)
y_value = np.linspace(-0.1,1.1,256)

def origin_f(x,y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2

def f(x,y,t):
    return t * (1 - x) ** 2 + 100 * t * (y - x * x) ** 2 - np.log(x+2.5) - np.log(y-0.5)

def df(x,y,t):
    gradient = np.array([[400 * t * x * x * x + 2 * t * x - 2 * t - 400 * t * x * y - (1 / (x + 2.5))],
                        [200 * t * y - 200 * t * x * x - (1 / (y - 0.5))]])
    return gradient

def ddf(x,y,t):
    Hessian = np.array([[1200 * t * x * x + 2 * t - 400 * t * y + (1 / (x * x + 5 * x + 6.25)), -400 * t * x ],
                        [-400 * t * x, 200 * t + (1 / (y * y - y + 0.25))]])
    return Hessian

def KKTMatrix(x,y,t):
    KKTMatrix = np.zeros((3,3))
    KKTMatrix[0,0] = ddf(x,y,t)[0,0]
    KKTMatrix[1,0] = ddf(x,y,t)[1,0]
    KKTMatrix[1,1] = ddf(x,y,t)[1,1]
    KKTMatrix[0,1] = ddf(x,y,t)[0,1]
    KKTMatrix[2,0] = 1
    KKTMatrix[2,1] = 1
    KKTMatrix[0,2] = 1
    KKTMatrix[1,2] = 1

    return KKTMatrix

def B(x,y,t):
    b = np.zeros((3, 1))
    b[0,0] = -df(x,y,t)[0,0]
    b[1,0] = -df(x,y,t)[1,0]
    b[2,0] = 2
    return b

def Newton(x,t):
    # 牛顿法
    while True:
        kkt_matrix = KKTMatrix(x[0, 0], x[1, 0],t)
        b_matrix = B(x[0, 0], x[1, 0],t)
        dx_nt = np.dot(np.linalg.inv(kkt_matrix), b_matrix)
        dx_nt = dx_nt[:2, ]
        lambda_2 = - df(x[0, 0], x[1, 0],t).T @ dx_nt

        if (lambda_2 / 2) <= epsilon:
            break

        # 回溯直线搜索
        while (f(x[0, 0] + t * dx_nt[0, 0], x[1, 0] + t * dx_nt[1, 0],t) > f(x[0, 0], x[1, 0],t) + alpha * t * np.dot(
                df(x[0, 0], x[1, 0],t).T, dx_nt)):
            t = beta * t
            print(t)

        # 修改x
        x = x + t * dx_nt

    print(x)
    return x

# 生成网格数据
X, Y = np.meshgrid(x_value, y_value)
#显示等高线
plt.figure()
# 填充等高线的颜色, 8是等高线分为几部分
plt.contourf(X, Y, origin_f(X, Y), 5, alpha=0)
# 绘制等高线
C = plt.contour(X, Y, origin_f(X, Y), 8, locator=ticker.LogLocator(), colors='black', linewidth=0.01)
# 绘制等高线数据
plt.clabel(C, inline=True, fontsize=10)

# 等式约束： x1 + x2 = 2
# 不等式约束: x1 >= -2.5, x2 >= 0.5

alpha = 0.09
beta = 0.8
epsilon = 0.00001 #误差阈值
x = np.array([[1.1],[0.9]])#初始点为(0.5,0.5),严格初始点

#用于存储，并显示轨迹
xv = [x[0,0]]
yv = [x[1,0]]
#作图，画出初始点
plt.plot(x[0, 0], x[1, 0], marker='o')

t = 0.1 #设置t0，满足大于0
miu = 1.1 #设置miu,满足大于1

k = 0

while True:
    x = Newton(x,t)

    xv.append(x[0, 0])
    yv.append(x[1, 0])

    if (2 / t) < epsilon:
        break
    t = miu * t
    k = k + 1
    print("k: ",k)



plt.plot(xv,yv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton\'s Method for Rosenbrock Function (Backtracking Line Search)')
plt.legend()
plt.show()
