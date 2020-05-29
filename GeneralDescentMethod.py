import scipy as sp
import numpy as np
import math
import matplotlib.pyplot as plt

x = np.arange(-20,20,0.1)
def f(x):
    return x * x

plt.plot(x,f(x))

alpha = 0.3
beta = 0.8
x = 19
eta = 0.01
descent = 2 * x
t_list = []

while (abs(descent) > eta):
    #确定下降方向
    if x > 0:
        dx = -1
    else:
        dx = 1
    #回溯直线搜索
    t = 1
    t_list.append(t)
    while (f(x+t*dx) > (f(x) + alpha * t * descent * dx)):
        t = beta * t
        t_list.append(t)

    #作图
    xx = np.arange(x-10,x+10,1)
    plt.plot(xx, f(x)+alpha*t*descent*(xx-x),color='red')

    # 修改x
    x = x + t * dx
    descent = 2 * x

plt.show()
