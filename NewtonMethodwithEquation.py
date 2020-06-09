import numpy as np
import matplotlib.pyplot as plt

#等式约束条件下的牛顿法（Rosenbrock函数）

# 定义x, y, 数据数量为256
from matplotlib import ticker

x_value = np.linspace(-1,1.1,256)
y_value = np.linspace(-0.1,1.1,256)

def f(x,y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2

def df(x,y):
    gradient = np.array([[400 * x * x * x + 2 * x - 2 - 400 * x * y],
                        [200 * y - 200 * x * x]])
    return gradient

def ddf(x,y):
    Hessian = np.array([[1200 * x * x + 2 - 400 * y, -400 * x ],
                        [-400 * x, 200]])
    return Hessian

def KKTMatrix(x,y):
    #print("ddf(x,y):",np.shape(ddf(x,y)))
    #print("np.ones(2,1).T:", np.shape(np.ones((2,1)).T))
    #print("np.ones(2,1):", np.shape(np.ones((2,1))))
    KKTMatrix = np.zeros((3,3))
    KKTMatrix[0,0] = ddf(x,y)[0,0]
    KKTMatrix[1,0] = ddf(x,y)[1,0]
    KKTMatrix[1,1] = ddf(x,y)[1,1]
    KKTMatrix[0,1] = ddf(x,y)[0,1]
    KKTMatrix[2,0] = 1
    KKTMatrix[2,1] = 1
    KKTMatrix[0,2] = 1
    KKTMatrix[1,2] = 1

    #KKTMatrix = np.array([[ddf(x,y), np.ones((2,1)).T ],[np.ones((2,1)), 0]])
    #print("KKTMatrix: ",np.shape(KKTMatrix))
    #print(KKTMatrix)
    #print("KKT系数矩阵行列式：",np.linalg.det(KKTMatrix))
    return KKTMatrix

def B(x,y):
    b = np.zeros((3, 1))
    print("aaa:",np.shape(-df(x,y)))
    #b = np.array([[-df(x,y)],[0]])
    b[0,0] = -df(x,y)[0,0]
    b[1,0] = -df(x,y)[1,0]
    b[2,0] = 0.8
    #print("b: ", np.shape(b))
    #print(b)
    return b

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
x = np.array([[0.2],[0.6]]) #随机选取一个起始点

epsilon = 0.0001 #误差阈值

#判断初始点是否满足 x = 1，不满足则初始该值进行修改


#用于存储，并显示轨迹
xv = [x[0,0]]
yv = [x[1,0]]
#作图，画出初始点
plt.plot(x[0, 0], x[1, 0], marker='o')
#plt.show()

t_list = []

while True:

    kkt_matrix = KKTMatrix(x[0,0],x[1,0])

    b_matrix = B(x[0,0],x[1,0])

    dx_nt = np.dot(np.linalg.inv(kkt_matrix),b_matrix)
    print(dx_nt)
    dx_nt = dx_nt[:2,]

    #lambda_2 = df(x[0,0],x[1,0]).T @ np.linalg.inv(ddf(x[0,0],x[1,0])) @ df(x[0,0],x[1,0])
    lambda_2 = - df(x[0,0],x[1,0]).T @ dx_nt
    #print(lambda_2)

    if (lambda_2 / 2) <= epsilon:
        break

    #回溯直线搜索
    t = 1
    t_list.append(t)

    f_x_t_delta_x = f(x[0,0]+t*dx_nt[0,0],x[1,0]+t*dx_nt[1,0])
    print("f_x_t_delta_x",f_x_t_delta_x)
    fx = f(x[0,0],x[1,0])
    print("fx", fx)
    #print(np.dot(df(x[0,0],x[1,0]).T, dx))
    f_x_a_t_grad_delta_x = fx + alpha * t * np.dot(df(x[0,0],x[1,0]).T, dx_nt)
    print("f_x_a_t_grad_delta_x", f_x_a_t_grad_delta_x)

    while (f(x[0,0]+t*dx_nt[0,0],x[1,0]+t*dx_nt[1,0]) > f(x[0,0],x[1,0]) + alpha * t * np.dot(df(x[0,0],x[1,0]).T, dx_nt)):
        t = beta * t
        t_list.append(t)

    # 修改x
    x = x + t * dx_nt
    xv.append(x[0, 0])
    yv.append(x[1, 0])

    #descent = 2 * x

plt.plot(xv,yv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton\'s Method for Rosenbrock Function')
plt.legend()
plt.show()