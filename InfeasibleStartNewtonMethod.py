import numpy as np
import matplotlib.pyplot as plt

#等式约束条件下的不可行初始点牛顿法（Rosenbrock函数）

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

def B(x,v):
    A = np.ones((1,2))
    AT = np.array([[1],[1]])
    X = np.array([[x[0,0]],[x[1,0]]])
    b = np.zeros((3, 1))
    #print("aaa:",np.shape(-df(x,y)))
    #b = np.array([[-df(x,y)],[0]])
    #print("AT@vAT@vAT@vAT@vAT@vAT@v")
    #print(AT@v)
    ATV = AT@v
    b[0,0] = df(x[0,0],x[1,0])[0,0] + ATV[0,0]
    b[1,0] = df(x[0,0],x[1,0])[1,0] + ATV[1,0]
    b[2,0] = A@X - 2
    #print("b: ", np.shape(b))
    #print("bbbbbbbbbbbbbbbbbbbbbbb")
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
beta = 0.9
x = np.array([[0.0],[0.0]]) #随机选取一个起始点
epsilon = 0.0001 #误差阈值
#等式系数矩阵A
A = np.ones((1,2))
AT = np.array([[1],[1]])
print(np.shape(AT))
print(np.shape(df(x[0,0],x[1,0])))
#计算v*
v = - A @ df(x[0,0],x[1,0]) * 0.5
#print("VVVVVVVVVVVVVVVVVVVVV")
#print(v)
vv = [v]
#用于存储，并显示轨迹
xv = [x[0,0]]
yv = [x[1,0]]
#作图，画出初始点
plt.plot(x[0, 0], x[1, 0], marker='o')
#plt.show()

t_list = []
f_list = []

while True:

    kkt_matrix = KKTMatrix(x[0,0],x[1,0])

    b_matrix = B(x,v)

    dx_w = np.linalg.inv(kkt_matrix) @ (-b_matrix)
    #print("$$"*40)
    #print(dx_w)
    #变量的个数 - 有效方程组的个数 = 对偶变量的个数
    dx_nt = dx_w[:2,]
    dv_nt = dx_w[2]

    #回溯直线搜索
    r_norm = np.linalg.norm(b_matrix, ord=2)
    t = 1
    t_list.append(t)

    #x_plus_t_dxnt = x + t * dx_nt
    #print("x_plus_t_dxnt")
    #print(x_plus_t_dxnt)
    #v_plus_t_dvnt = v + t * dv_nt
    #print("v_plus_t_dvnt")
    #print(v_plus_t_dvnt)

    #twice_r = B(x_plus_t_dxnt,v_plus_t_dvnt)
    #twice_r_norm = np.linalg.norm(twice_r,ord=2)

    while (np.linalg.norm(B(x + t * dx_nt,v + t * dv_nt),ord=2) > ((1-alpha*t) * r_norm)):
        t = beta * t
        t_list.append(t)
        print("ttttttttttttttttttttttttt")
        print(t)

    # 修改x
    x = x + t * dx_nt
    v = v + t * dv_nt
    xv.append(x[0, 0])
    yv.append(x[1, 0])
    vv.append(v)
    f_list.append(f(x[0, 0], x[1, 0]))
    #descent = 2 * x

    if (A@x == 2) and (r_norm <= epsilon):
        break

plt.plot(xv,yv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton\'s Method for Rosenbrock Function')
plt.legend()
plt.show()

plt.plot(yv, color='red',label='x2')
plt.plot(xv, color='blue',label='x1')
plt.xlabel('k')
plt.ylabel('X')
plt.grid()
plt.legend()
plt.show()


plt.plot(f_list, color='orange',label='f(x,y)')
plt.xlabel('k')
plt.ylabel('f(x,y)')
plt.grid()
plt.legend()
plt.show()