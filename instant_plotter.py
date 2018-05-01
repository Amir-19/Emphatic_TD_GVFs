import numpy as np
import matplotlib.pyplot as plt

d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
d6 = []
d7 = []
d8 = []
be = []
ve = []

ed1 = []
ed2 = []
ed3 = []
ed4 = []
ed5 = []
ed6 = []
ed7 = []
ed8 = []
ebe = []
eve = []

s1_ds = np.loadtxt('gtd_data.txt')
s2_ds = np.loadtxt('etd_data.txt')
for i in range(s1_ds.shape[0]):
    d1.append(s1_ds[i][1])
    d2.append(s1_ds[i][2])
    d3.append(s1_ds[i][3])
    d4.append(s1_ds[i][4])
    d5.append(s1_ds[i][5])
    d6.append(s1_ds[i][6])
    d7.append(s1_ds[i][7])
    d8.append(s1_ds[i][8])
    ve.append(s1_ds[i][9])
    be.append(s1_ds[i][10])

    ed1.append(s2_ds[i][1])
    ed2.append(s2_ds[i][2])
    ed3.append(s2_ds[i][3])
    ed4.append(s2_ds[i][4])
    ed5.append(s2_ds[i][5])
    ed6.append(s2_ds[i][6])
    ed7.append(s2_ds[i][7])
    ed8.append(s2_ds[i][8])
    eve.append(s2_ds[i][9])
    ebe.append(s2_ds[i][10])

    #d2.update(i, [s2_ds[i][1],s2_ds[i][2],s2_ds[i][3],s2_ds[i][4],s2_ds[i][5],s2_ds[i][6],s2_ds[i][7],s2_ds[i][8],s2_ds[i][9],s2_ds[i][10]])
g1 = plt.figure(1)
g1.set_size_inches(8.5, 6.5, forward=True)
plt.suptitle('GTD Weights and Error')
plt.plot(d1)
plt.plot(d2)
plt.plot(d3)
plt.plot(d4)
plt.plot(d5)
plt.plot(d6)
plt.plot(d7)
plt.plot(d8)
plt.plot(ve)
plt.plot(be)
plt.xlabel('Steps')
plt.ylabel('Value')
plt.legend(['w1', 'w2', 'w3', 'w4','w5','w6','w7','w8','RMSVE','RMSPBE'], loc='upper left')
g1.show()


g2 = plt.figure(2)
g2.set_size_inches(8.5, 6.5, forward=True)
plt.suptitle('ETD Weights and Error')
plt.plot(ed1)
plt.plot(ed2)
plt.plot(ed3)
plt.plot(ed4)
plt.plot(ed5)
plt.plot(ed6)
plt.plot(ed7)
plt.plot(ed8)
plt.plot(eve)
plt.plot(ebe)
plt.xlabel('Steps')
plt.ylabel('Value')
plt.legend(['w1', 'w2', 'w3', 'w4','w5','w6','w7','w8','RMSVE','RMSPBE'], loc='upper left')
g2.show()
raw_input()
