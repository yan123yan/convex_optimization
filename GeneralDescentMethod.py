import scipy as sp
import numpy as np
import math
import matplotlib.pyplot as plt

x = np.arange(-20,20,0.1)

def f(x):
    return x ** 2

plt.plot(x,f(x))

def df(x):
    descent = 2 * (x ** 1)
    return descent

def  tangent_func(x,x_0):
    return 2 * x_0 * (x - x_0) + f(x_0)

alpha = 0.3
beta = 0.8
x = 20 #随机选取一个起始点
eta = 0.01 #学习率

t_list = []
history_x=[x]

while (abs(df(x)) > eta):
    #确定下降方向
    if x > 0:
        dx = -1
    else:
        dx = 1
    #回溯直线搜索
    t = 1
    t_list.append(t)
    while (f(x+t*dx) > (f(x) + alpha * t * df(x) * dx)):
        t = beta * t
        t_list.append(t)

    #作图
    xx = np.arange(x-10,x+10,1)
    #plt.plot(xx, f(x)+alpha*t*df(x)*(xx-x),color='pink')
    plt.plot(xx, tangent_func(xx, x), color='orange')

    # 修改x
    x = x + t * dx
    history_x.append(x)
    #descent = 2 * x

print(history_x)
print(type(history_x))
history_x.reverse()
plt.plot(np.array(history_x),f(np.array(history_x)),color="red",marker='*')
plt.show()