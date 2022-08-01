import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import togif
import os
import time
from problem import Func

def getB(T):
    B = []
    data = pd.read_csv('vector.csv', sep=',')
    df = pd.DataFrame(data)
    # calculate Euclidean distance
    for r in range(df.shape[0]):
        b = []
        l = []
        for i in range(M):
            l.append(df.values[r, i])
        for r1 in range(df.shape[0]):
            l1 = []
            
            for j in range(M):
                l1.append(df.values[r1, j])
            sum = 0
            for k in range(M):
                sum += (l[k] - l1[k]) ** 2
            b.append(float('%.4f' % sum))
        B.append(b)
    # sort
    B = np.array(B)
    B = np.array(B.argsort(axis=1)[:, 0: T])
    # data = pd.read_csv('vector.csv', sep='\t', header=None)
    # print(data, type(data))
    # B = data
    # B = B - 1
    # data = pd.read_csv('v.csv', sep=',')
    # df = data
    return B, df, len(B)

def getboundary1(h):
    if h >= 0 and h < 15.9:
        return [125.8, 44]
    elif h >= 15.9 and h <= 32.4:
        return [125.8, 40 - (4/(75-15.9))* (h-75)]
    elif h >= 32.4 and h < 75:
        return [125.8 + (125.8 - 110.2)/(32.4 - 135.6) * (h-32.4), 40 - (4/(75-15.9))* (h-75)]
    elif h >= 75 and h <= 135.6:
        return [125.8 + (125.8 - 110.2)/(32.4 - 135.6) * (h-32.4), 40 + (110.2-40)/(135.6-75) * (h-75)]
    else:
        return [0, 0]

def getboundary2(h):
    if h >= 0 and h < 40:
        return [60 + 15/(-40) * h, 20 + 10 / (-40)*h]
    elif h >= 40 and h <= 55:
        return [60 + 15/(-40) * h, 10 + (35)/15 * (h-40)]
    else:
        return [0, 0]

def getboundary3(h):
    if h >= 0 and h <= 20:
        return [105 + 15/-25*h, 35]
    elif h >20 and h <= 25:
        return [105 + 15/-25*h, 35 + 55/25*(h-20)]
    elif h > 25 and h <= 45:
        return [90, 35 + 55/25*(h-20)]
    else:
        return [0, 0]


def create_pop(N, D):
    p1max = 135
    p1min = 35
    h5max = 60
    h5min = 0
    boundary = [[135, 125.8, 135.6, 60, 55, 105, 45, 60], [35, 40, 0, 10, 0, 35, 0, 0]]

    population = []
    outboundarylist = []
    for _ in range(N):
        pri = []
        pri.append(np.random.rand())
        pri[-1] = pri[-1] * p1max + (1-pri[-1]) * p1min

        h = np.random.rand() * (boundary[0][2] - boundary[1][2]) + boundary[1][2]
        bh = getboundary1(h)
        p = np.random.rand() * (bh[0] - bh[1]) + bh[1]
        pri.append(p)
        pri.append(h)

        h = np.random.rand() * (boundary[0][4] - boundary[1][4]) + boundary[1][4]
        bh = getboundary2(h)
        p = np.random.rand() * (bh[0] - bh[1]) + bh[1]
        pri.append(p)
        pri.append(h)

        h = np.random.rand() * (boundary[0][6] - boundary[1][6]) + boundary[1][6]
        bh = getboundary3(h)
        p = np.random.rand() * (bh[0] - bh[1]) + bh[1]
        pri.append(p)
        pri.append(h)

        pri.append(np.random.rand())
        pri[-1] = pri[-1] * (h5max - h5min) + h5min
        pri = np.array(pri)
        amountp = np.sum(pri[pindex])
        amounth = np.sum(pri[hindex])
        pri[pindex] *= psum / amountp
        pri[hindex] *= hsum / amounth
        population.append(pri)
        f = 0

        h = pri[2]
        bh = getboundary1(h)
        if bh[0] == 0 and bh[1] == 0:
            f = 1
        elif pri[1] > bh[0] or pri[1] < bh[1]:
            f = 1

        h = pri[4]
        bh = getboundary2(h)
        if bh[0] == 0 and bh[1] == 0:
            f = 1
        elif pri[3] > bh[0] or pri[3] < bh[1]:
            f = 1

        h = pri[6]
        bh = getboundary3(h)
        if bh[0] == 0 and bh[1] == 0:
            f = 1
        elif pri[5] > bh[0] or pri[5] < bh[1]:
            f = 1


        if f == 0:
            outboundarylist.append(False)
        else:
            outboundarylist.append(True)
    output = population
    return output, np.array(boundary), outboundarylist

def evolution(GEN, pop, boundary):
    global z
    for round in range(GEN):
        print(f'\r第{round}轮', end='')
        for i in range(N):
            if np.random.rand() < 0.9:
                P = B[i]
            else:
                P = np.arange(N)
            k = list(range(len(P)))
            random.shuffle(k)
            Offspring = gen(pop[i], pop[P[k[0]]], pop[P[k[1]]], boundary)
            f = 1
            # for b in range(1, 4):
            #     pb = getboundary(Offspring[2 * b])
            #     if not(Offspring[2*b-1] > pb[1] and Offspring[2*b-1] < pb[0]):
            #         f = 0
            #         break

            pb = getboundary1(Offspring[2])
            if not(Offspring[1] > pb[1] and Offspring[1] < pb[0]):
                f = 0
            pb = getboundary2(Offspring[4])
            if not(Offspring[3] > pb[1] and Offspring[3] < pb[0]):
                f = 0
            pb = getboundary3(Offspring[6])
            if not(Offspring[5] > pb[1] and Offspring[5] < pb[0]):
                f = 0

            if f == 1:
                penalty = 0
            else:
                penalty = np.inf
            Offfuncvalue = np.array(Func(Offspring))
            Offfuncvalue[0] /= equilibriumf1
            Offfuncvalue[1] /= equilibriumf2

            # z = np.array([np.min([z[m], Offfuncvalue[m]]) for m in range(M)])
            # z = np.array([0, 0])
            # 加权和法

            if np.sum(Offfuncvalue * vectors[i]) + penalty < np.sum(functionvalue[i] * vectors[i]):
                pop[i] = Offspring
                functionvalue[i] = Offfuncvalue
            
            # 切比雪夫方法
            # if np.max(abs(Offfuncvalue) * vectors[i]) + penalty < np.max(abs(functionvalue[i]) * vectors[i]):
            #     pop[i] = Offspring
            #     functionvalue[i] = Offfuncvalue

            c = 0
            l = list(range(len(P)))
            random.shuffle(l)
            for j in l:
                if c >= nr:
                    break

                g_old = np.sum(functionvalue[P[j]] * vectors[P[j]])

                g_new = np.sum(Offfuncvalue * vectors[P[j]]) + penalty
                # g_old = np.max(abs(functionvalue[P[j]]) * vectors[P[j]])
                # g_new = np.max(abs(Offfuncvalue) * vectors[P[j]]) + penalty


                if g_new < g_old:
                    pop[P[j]] = Offspring
                    functionvalue[P[j]] = Offfuncvalue
                    c += 1

        plt.figure(figsize=(10, 10), dpi=120)
        plt.scatter(functionvalue[:, 0], functionvalue[:, 1])
        plt.scatter(base_dot[0], base_dot[1], c=['red', 'red'])
        plt.title('Round {}'.format(str(round + 1)))
        plt.xlabel('Emission/kg', fontdict={'size': 25})
        plt.ylabel('Cost/1e4 USD', fontdict={'size': 25})
        plt.savefig('./gif/{}.png'.format(str(round + 1)))
        plt.close()


def mutate(Offspring, boundary):
    ProM = 1 / D
    DisM = 5
    max = boundary[0]
    min = boundary[1]

    k = np.random.rand(D)
    t = k <= ProM
    miu = np.random.rand(D)
    t1 = miu < 0.5

    temp = t & t1
    Offspring[temp] = Offspring[temp] + (max[temp] - min[temp]) * ((2 * miu[temp] + (1 - 2 * miu[temp]) * (
                1 - (Offspring[temp] - min[temp]) / (max[temp] - min[temp])) ** (DisM + 1)) ** (1 / (DisM + 1)) - 1)

    t1 = ~t1
    temp = t & t1
    Offspring[temp] = Offspring[temp] + (max[temp] - min[temp]) * (1 - (2 * (1 - miu[temp]) + 2 * (miu[temp] - 0.5) * (
                1 - (max[temp] - Offspring[temp]) / (max[temp] - min[temp])) ** (DisM + 1)) ** (1 / (DisM + 1)))
    Offspring[Offspring > max] = max[Offspring > max]
    Offspring[Offspring < min] = min[Offspring < min]
    amountp = np.sum(Offspring[pindex])
    amounth = np.sum(Offspring[hindex])
    Offspring[pindex] *= (psum / amountp)
    Offspring[hindex] *= (hsum / amounth)



    return Offspring




def gen(r1, r2, r3, boundary):#变异
    D = len(r1)
    CR = 1
    F = 0.5
    Offspring = r1
    # print('r1')
    # print(r1)
    temp = np.random.rand(D) <= CR
    Offspring[temp] = Offspring[temp] + F * (r2[temp] - r3[temp])

    Offspring = mutate(Offspring, boundary)


    return np.array(Offspring)


gx = -1

def rm_dir():
    fl = os.listdir('./gif')
    fl = [int(i.split('.')[0]) for i in fl]
    fl.sort()
    image_list = ['./gif/' + str(i) + '.png' for i in fl]
    for i in image_list:
        os.remove(i)

if __name__=='__main__':
    rm_dir()
    equilibriumf1 = 1
    equilibriumf2 = 10000
    base_dot = [[7.5/equilibriumf1, 5.1/equilibriumf1], [14504.2/equilibriumf2, 15137.3/equilibriumf2]]
    np.set_printoptions(threshold=np.inf)
    T = 80 # 邻居数量
    D = 8 # 变量数量
    GEN = 10000 #迭代次数
    N = 1000 # 种群规模
    M = 2 # 函数个数
    delta = 0.9
    nr = 16
    res = getB(T)
    B = np.array(res[0]) #邻居权重向量
    pindex = [0, 1, 3, 5] #p变量索引
    hindex = [2, 4, 6, 7] #h变量索引
    psum = 300
    hsum = 150

    vectors = np.array(res[1]) #权向量
    res = create_pop(N, D) #种群
    pop = res[0]
    boundary = res[1]
    outboundarylist = np.array(res[2], dtype=bool)
    functionvalue = [Func(i) for i in pop] #函数值
    functionvalue = np.array(functionvalue)
    functionvalue[outboundarylist] = np.array([np.inf, np.inf])
    for i in range(N):
        functionvalue[i][0] /= equilibriumf1
        functionvalue[i][1] /= equilibriumf2
    plt.figure(figsize=(10, 10), dpi=120)
    plt.scatter(functionvalue[:, 0], functionvalue[:, 1])
    plt.scatter(base_dot[0], base_dot[1], c=['red', 'red'])
    plt.savefig('./gif/0.png')
    plt.show()
    # z = np.array([np.min(functionvalue[:, i]) for i in range(M)]) #理想点
    z = np.zeros(M)
    evolution(GEN, pop, boundary)
    # togif.main()



    plt.figure(figsize=(10, 10), dpi=120)
    plt.scatter(functionvalue[:, 0], functionvalue[:, 1])
    plt.scatter(base_dot[0], base_dot[1], c=['red', 'red'])
    plt.show()
    plt.close()
    f = open('res.csv', 'w')
    for i in range(N):
        f.write(str(vectors[i]) + ',' + str(functionvalue[i]) + ',' + str(pop[i]).replace('\n', '\t') + '\n')
    f.close()