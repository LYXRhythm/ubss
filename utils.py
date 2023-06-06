import numpy as np
import matplotlib.pyplot as plt

def display(s, x, s_hat, signal_length=200):
    t = np.arange(signal_length)

    s_num = s.shape[0]
    x_num = x.shape[0]
    s_hat_num = s_hat.shape[0]
    num = max(s_num, x_num, s_hat_num)

    print(s_num)
    print(x_num)
    print(s_hat_num)
    ax_s = []
    for i1 in range(s_num):
        ax_s.append(plt.subplot(3, num, i1+1))
    for i1 in range(s_num):
        ax_s[i1].plot(t, s[i1])

    ax_x = []
    for i2 in range(x_num):
        ax_x.append(plt.subplot(3, num, num+i2+1))
    for i2 in range(x_num):
        ax_x[i2].plot(t, x[i2])

    ax_s_hat = []
    for i3 in range(s_hat_num):
        ax_s_hat.append(plt.subplot(3, num, num+num+i3+1))
    for i3 in range(s_hat_num):
        ax_s_hat[i3].plot(t, s_hat[i3])

    plt.show()