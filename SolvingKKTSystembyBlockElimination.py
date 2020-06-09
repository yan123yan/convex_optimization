import numpy as np
import matplotlib.pyplot as plt

#等式约束条件下的不可行初始点牛顿法（Rosenbrock函数）[使用消元法求解KKT系统]

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

    KKTMatrix = np.zeros((3,3))
    KKTMatrix[0,0] = ddf(x,y)[0,0]
    KKTMatrix[1,0] = ddf(x,y)[1,0]
    KKTMatrix[1,1] = ddf(x,y)[1,1]
    KKTMatrix[0,1] = ddf(x,y)[0,1]
    KKTMatrix[2,0] = 1
    KKTMatrix[2,1] = 1
    KKTMatrix[0,2] = 1
    KKTMatrix[1,2] = 1

    return KKTMatrix

def B(x,v):
    A = np.ones((1,2))
    AT = np.array([[1],[1]])
    X = np.array([[x[0,0]],[x[1,0]]])
    b = np.zeros((3, 1))
    ATV = AT @ v
    b[0,0] = df(x[0,0],x[1,0])[0,0] + ATV[0,0]
    b[1,0] = df(x[0,0],x[1,0])[1,0] + ATV[1,0]
    b[2,0] = A@X - 2
    return b

def BlockElimination(x,B):
    #初始值定义
    A = np.ones((1, 2))
    AT = np.array([[1], [1]])
    #Step 1
    #计算H-1AT
    H_inv = np.linalg.inv(ddf(x[0,0],x[1,0]))
    H_inv_AT = H_inv @ AT
    print("H_inv_AT")
    print(H_inv_AT)
    #计算H-1g
    g = B[:2,]
    H_inv_g = H_inv @ g
    print("H_inv_g")
    print(H_inv_g)
    #Step 2
    #计算Schur补 S
    S = - A @ H_inv @ AT
    print("S")
    print(S)
    S_inv = np.linalg.inv(S)
    print("S_inv")
    print(S_inv)

    #Step 3
    h = B[2,]
    w = S_inv @ (A @ H_inv_g - h)

    #Step 4
    v = H_inv @ (- AT @ w - g)

    print("w")
    print(w)
    print("v")
    print(v)

    return v, w

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
x = np.array([[0],[0]]) #随机选取一个起始点
epsilon = 0.0001 #误差阈值
#等式系数矩阵A
A = np.ones((1,2))
AT = np.array([[1],[1]])
print(np.shape(AT))
print(np.shape(df(x[0,0],x[1,0])))
#计算v*
v = - A @ df(x[0,0],x[1,0]) * 0.5
vv = [v]
#用于存储，并显示轨迹
xv = [x[0,0]]
yv = [x[1,0]]
#作图，画出初始点
plt.plot(x[0, 0], x[1, 0], marker='o')
#plt.show()

t_list = []

while True:

    dx_nt, dv_nt = BlockElimination(x,B(x,v))

    #回溯直线搜索
    r_norm = np.linalg.norm(B(x,v), ord=2)
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
    #descent = 2 * x

    if (A@x == 2) and (r_norm <= epsilon):
        break

plt.plot(xv,yv,label='track')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton\'s Method for Rosenbrock Function')
plt.legend()
plt.show()