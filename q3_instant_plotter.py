import numpy as np
import matplotlib.pyplot as plt

d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
d6 = []
d7 = []


s1_ds = np.loadtxt('q3_data.txt')

for i in range(s1_ds.shape[0]):
    d1.append(s1_ds[i][1])
    d2.append(s1_ds[i][2])
    d3.append(s1_ds[i][3])
    d4.append(s1_ds[i][4])
    d5.append(s1_ds[i][5])
    d6.append(s1_ds[i][6])
    d7.append(s1_ds[i][7])


    #d2.update(i, [s2_ds[i][1],s2_ds[i][2],s2_ds[i][3],s2_ds[i][4],s2_ds[i][5],s2_ds[i][6],s2_ds[i][7],s2_ds[i][8],s2_ds[i][9],s2_ds[i][10]])
g1 = plt.figure(1)
g1.set_size_inches(19.5, 4.5, forward=True)
plt.suptitle('Off-policy question data and predictions')
plt.plot(d1)
plt.plot(d2)
plt.plot(d3)
plt.plot(d4)
plt.plot(d5)


plt.xlabel('Time_Step')
plt.ylabel('Value')
plt.legend(['servo 1 ang * 10', 'GTD Prediction', 'ETD Prediction', 'Gamma*3','Cumulant*6'], loc='upper left')
g1.show()


g2 = plt.figure(2)
plt.suptitle('Off-policy question TD Error')
g2.set_size_inches(19.5, 4.5, forward=True)
plt.plot(d6)
plt.plot(d7)
plt.xlabel('Time_Step')
plt.ylabel('Value')
plt.legend(['GTD error', 'ETD error'], loc='upper left')
g2.show()
raw_input()
