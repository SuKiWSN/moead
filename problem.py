import numpy as np

pd = 300# MW
hd = 150# MWH

def f1(x):
    p1 = x[0]
    c1 = 254.8863 + 7.6997*p1 + 0.00172*(p1**2) + 0.000115 * (p1**3)
    e1 = 1e-4 * (4.091 - 5.554*p1 + 6.490*(p1**2)) + 2 * 1e-4 * np.exp(0.02857 * p1)
    return np.array([e1, c1])

def f2(x):
    p2 = x[1]
    h2 = x[2]
    c2 = 1250 + 36 * p2 + 0.0435*(p2**2) + 0.6 * h2 + 0.027 * (h2 ** 2) + 0.011 * p2 * h2
    e2 = 0.00165 * p2
    return np.array([e2, c2])

def f3(x):
    p3 = x[3]
    h3 = x[4]
    c3 = 2650 + 34.5 * p3 + 0.1035 * (p3 ** 2) + 2.203 * h3 + 0.025 * (h3 ** 2) + 0.051 * p3 * h3
    e3 = 0.0022 * p3
    return np.array([e3, c3])

def f4(x):
    p4 = x[5]
    h4 = x[6]
    c4 = 1565 + 20 * p4 + 0.072 * (p4 ** 2) + 2.3 * h4 + 0.02 * (h4 ** 2) + 0.04*p4*h4
    e4 = 0.0011 * p4
    return np.array([e4, c4])

def f5(x):
    h5 = x[7]
    c5 = 950 + 2.0109*h5 + 0.038 * (h5 ** 2)
    e5 = 0.0017 * h5
    return np.array([e5, c5])

def Func(x):
    r1 = f1(x)
    r2 = f2(x)
    r3 = f3(x)
    r4 = f4(x)
    r5 = f5(x)
    return r1 + r2 + r3 + r4 + r5

