import numpy as np
import matplotlib.pyplot as plt

O = (0, 0)
x = np.arange(0, 1.00, 0.001)
x[0] = 1e-5
y = 1 - x
y[-1] = 1e-5
c = zip(x, y)
f = open('vector.csv', 'w')
f.write('0,1' + '\n')
for i in c:
    print(i)
    f.write(str(i)[1: -1] + '\n')

for i in range(len(x)):
    plt.plot([0, x[i]], [0, y[i]])
plt.show()