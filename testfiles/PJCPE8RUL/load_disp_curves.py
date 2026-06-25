import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def mm2inch(x):
    return x / 25.4

plt.style.use('rcparams.mplstyle')
fig, ax = plt.subplots(1, 1, figsize=(mm2inch(95), mm2inch(95)))#, sharey=True)
#linestyles = ['-', '-', '-']
linestyles = ['-', '--', '-.']
colors = plt.cm.magma(np.linspace(0.0, .85, 3 ))[::1]




#load csv
rf_U = np.loadtxt('rf_cpe8ul_med.csv', delimiter=' ')
u_U = np.loadtxt('u_cpe8ul_med.csv', delimiter=' ')

rf_U_r = np.loadtxt('rf_cpe8rul_med.csv', delimiter=' ')
u_U_r = np.loadtxt('u_cpe8rul_med.csv', delimiter=' ')

rf_PJ = np.loadtxt('rf_pjcpe8ul_med.csv', delimiter=' ')
u_PJ = np.loadtxt('u_pjcpe8ul_med.csv', delimiter=' ')

#plot
ax.plot(u_U[:,0], rf_U[:,1], label='standard/full',color=colors[0])
ax.plot(u_U_r[:,0], rf_U_r[:,1], label='standard/reduced',color=colors[1])
ax.plot(u_PJ[:,0], rf_PJ[:,1], label='mixed',color=colors[2])
ax.plot(u_U[-1,0], rf_U[-1,1], 'x',color=colors[0])
ax.plot(u_PJ[-1,0], rf_PJ[-1,1], 'x',color=colors[2])
ax.plot(u_U_r[-1,0], rf_U_r[-1,1], 'x',color=colors[1])
ax.set_xlim(1,2)
ax.set_xlabel('displacement (mm)')
ax.set_ylabel('force (N)')
ax.grid()
plt.legend()
plt.show()

