import numpy as np
import matplotlib.pyplot as plt

#无约束条件下的牛顿法（Rosenbrock函数）

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

def ddf(x,y):
    Hessian = np.array([[1200 * x * x + 2 - 400 * y, -400 * x ],
                        [-400 * x, 200]])
    return Hessian

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
epsilon = 0.0001 #误差阈值

#用于存储，并显示轨迹
xv = [x[0,0]]
yv = [x[1,0]]
#作图，画出初始点
plt.plot(x[0, 0], x[1, 0], marker='o')
#plt.show()

t_list = []
max_f_backtracking = 0
f_backtracking_list =[]
k_backtracking = 0

while True:
    #Hessian矩阵的逆 乘 梯度
    # dx_nt = - np.linalg.inv(ddf(x[0,0],x[1,0])) * df(x[0,0],x[1,0])
    dx_nt = - np.dot(np.linalg.inv(ddf(x[0, 0], x[1, 0])), df(x[0, 0], x[1, 0]))
    print(dx_nt)

    lambda_2 = df(x[0,0],x[1,0]).T @ np.linalg.inv(ddf(x[0,0],x[1,0])) @ df(x[0,0],x[1,0])
    print(lambda_2)

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
    f_backtracking_list.append(f(x[0, 0],x[1, 0]))
    #descent = 2 * x
    max_f_backtracking = f(x[0, 0], x[1, 0])
    k_backtracking = k_backtracking + 1

plt.plot(xv,yv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton\'s Method for Rosenbrock Function (Backtracking Line Search)')
plt.legend()
plt.show()









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
epsilon = 0.0001 #误差阈值

#用于存储，并显示轨迹
xxv = [x[0,0]]
yyv = [x[1,0]]
#作图，画出初始点
plt.plot(x[0, 0], x[1, 0], marker='o')
#plt.show()

t_list = []
max_f_exact = 0
k_exact = 0
f_exact_list =[]

while True:
    #Hessian矩阵的逆 乘 梯度
    # dx_nt = - np.linalg.inv(ddf(x[0,0],x[1,0])) * df(x[0,0],x[1,0])
    dx_nt = - np.dot(np.linalg.inv(ddf(x[0, 0], x[1, 0])), df(x[0, 0], x[1, 0]))

    lambda_2 = df(x[0,0],x[1,0]).T @ np.linalg.inv(ddf(x[0,0],x[1,0])) @ df(x[0,0],x[1,0])

    if (lambda_2 / 2) <= epsilon:
        break

    #精确直线搜索
    s = np.arange(0, 1, 0.001)
    ff_list = []
    for i in range(len(s)):
        re = x + s[i] * dx_nt
        ff = f(re[0, 0], re[1, 0])
        ff_list.append(ff)
    t = s[np.array(ff_list).argmin()]
    t_list.append(t)

    # 修改x
    x = x + t * dx_nt
    xxv.append(x[0, 0])
    yyv.append(x[1, 0])
    f_exact_list.append(f(x[0, 0],x[1, 0]))
    #descent = 2 * x
    max_f_exact = f(x[0, 0], x[1, 0])
    k_exact = k_exact + 1

plt.plot(xxv,yyv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton\'s Method for Rosenbrock Function (Exact Line Search)')
plt.legend()
plt.show()


plt.plot(f_exact_list, color='red',label='exact line search')
plt.plot(k_exact-1,max_f_exact,marker="*",color='red')
plt.plot(k_backtracking-1,max_f_backtracking,marker="o",color='blue')
plt.plot(f_backtracking_list, color='blue',label='backtracking line search')
plt.xlabel('K')
plt.ylabel('f(x^(k))-p^*')
plt.grid()
plt.legend()
plt.show()