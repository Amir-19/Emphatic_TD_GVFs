import numpy
from dynamic_plotter import *
import time


def main():

    d1 = DynamicPlot(window_x=100, title='GTD Weights and Error', xlabel='Steps', ylabel='Value')
    d1.add_line('w1')
    d1.add_line('w2')
    d1.add_line('w3')
    d1.add_line('w4')
    d1.add_line('w5')
    d1.add_line('w6')
    d1.add_line('w7')
    d1.add_line('w8')
    d1.add_line('RMSVE')
    d1.add_line('RMSPBE')
    d2 = DynamicPlot(window_x=100, title='ETD Weights and Error', xlabel='Steps', ylabel='Value')
    d2.add_line('w1')
    d2.add_line('w2')
    d2.add_line('w3')
    d2.add_line('w4')
    d2.add_line('w5')
    d2.add_line('w6')
    d2.add_line('w7')
    d2.add_line('w8')
    d2.add_line('RMSVE')
    d2.add_line('RMSPBE')

    # you need to choose the file you want to do the offline plot from
    s1_ds = numpy.loadtxt('gtd_data.txt')
    s2_ds = numpy.loadtxt('etd_data.txt')
    for i in range(s1_ds.shape[0]):
        print(len(s1_ds[i]))
        d1.update(i, [s1_ds[i][1],s1_ds[i][2],s1_ds[i][3],s1_ds[i][4],s1_ds[i][5],s1_ds[i][6],s1_ds[i][7],s1_ds[i][8],s1_ds[i][9],s1_ds[i][10]])
        d2.update(i, [s2_ds[i][1],s2_ds[i][2],s2_ds[i][3],s2_ds[i][4],s2_ds[i][5],s2_ds[i][6],s2_ds[i][7],s2_ds[i][8],s2_ds[i][9],s2_ds[i][10]])
        # change to make fast and slow plotting
        time.sleep(0.0)
    while True:
        pass
if __name__ == '__main__':
    main()