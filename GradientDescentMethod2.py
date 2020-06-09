import numpy as np
import matplotlib.pyplot as plt

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

# 生成网格数据
X, Y = np.meshgrid(x_value, y_value)

#显示3D图像
fig = plt.figure()
ax = plt.gca(projection='3d')
#ax.plot(X,Y,f(X,Y))
ax.plot_surface(X,Y,f(X,Y),cmap='jet')
#plt.show()

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
eta = 0.000001 #学习率

#print(df(x))

xv = [x[0,0]]
yv = [x[1,0]]
plt.plot(x[0, 0], x[1, 0], marker='o')
#plt.show()

t_list = []
f_backtracking_list = []
max_f_backtracking = 0
k_backtracking = 0

#采用回溯直线搜索的梯度下降
while (abs(df(x[0,0],x[1,0])[0,0]) > eta and abs(df(x[0,0],x[1,0])[1,0]) > eta):
    #确定下降方向
    dx = -df(x[0,0],x[1,0])
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

    # 修改x
    x = x + t * dx
    t_list.append(t)

    xv.append(x[0, 0])
    yv.append(x[1, 0])
    f_backtracking_list.append(f(x[0,0],x[1,0]))
    max_f_backtracking = f(x[0,0],x[1,0])
    k_backtracking = k_backtracking + 1
    #descent = 2 * x

plt.plot(xv,yv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graident Descent Method for Rosenbrock Function')
plt.legend()
plt.show()


plt.plot(yv, color='red',label='x2')
plt.plot(xv, color='blue',label='x1')
plt.xlabel('Times')
plt.ylabel('X')
plt.grid()
plt.legend()
plt.show()


plt.plot(f_backtracking_list, color='orange',label='f(x,y)')
plt.xlabel('Times')
plt.ylabel('f(x)')
plt.grid()
plt.legend()
plt.show()



fig = plt.figure()
ax = plt.gca(projection='3d')
#ax.plot(X,Y,f(X,Y))
ax.plot_surface(X,Y,f(X,Y),cmap='jet')
#plt.show()

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
eta = 0.000001 #学习率

#print(df(x))

xxv = [x[0,0]]
yyv = [x[1,0]]
plt.plot(x[0, 0], x[1, 0], marker='o')
#plt.show()

t_list = []
f_exact_list = []
max_f_exact = 0
k_exact = 0

#采用精确直线搜索的梯度下降
while (abs(df(x[0,0],x[1,0])[0,0]) > eta and abs(df(x[0,0],x[1,0])[1,0]) > eta):
    #确定下降方向
    dx = -df(x[0,0],x[1,0])

    #精确直线搜索

    s = np.arange(0, 1, 0.001)
    ff_list = []
    for i in range(len(s)):
        re = x + s[i] * dx
        ff = f(re[0,0],re[1,0])
        ff_list.append(ff)
    t = s[np.array(ff_list).argmin()]
    t_list.append(t)
    #print(t)

    # 修改x
    x = x + t * dx
    print(x)

    xxv.append(x[0, 0])
    yyv.append(x[1, 0])
    f_exact_list.append(f(x[0, 0], x[1, 0]))
    max_f_exact = f(x[0, 0], x[1, 0])
    k_exact = k_exact + 1
    #descent = 2 * x

plt.plot(xxv,yyv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graident Descent Method for Rosenbrock Function')
plt.legend()
plt.show()


plt.plot(yyv, color='red',label='x2')
plt.plot(xxv, color='blue',label='x1')
plt.xlabel('Times')
plt.ylabel('X')
plt.grid()
plt.legend()
plt.show()


plt.plot(f_exact_list, color='orange',label='f(x,y)')
plt.xlabel('Times')
plt.ylabel('f(x)')
plt.grid()
plt.legend()
plt.show()

plt.plot(f_exact_list, color='red',label='exact line search')
plt.plot(k_exact,max_f_exact,marker="*",color='red')
plt.plot(k_backtracking,max_f_backtracking,marker="o",color='blue')
plt.plot(f_backtracking_list, color='blue',label='backtracking line search')
plt.xlabel('K')
plt.ylabel('f(x^(k))-p^*')
plt.grid()
plt.legend()
plt.show()