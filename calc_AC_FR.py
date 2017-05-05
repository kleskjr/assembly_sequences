from matplotlib import pyplot as plt 
import numpy
import calc_spikes


dt = .1
ww=2.
t=numpy.arange(-3*ww,3*ww,dt)
gas = numpy.exp(-(t/ww)**2)
gas/=gas.sum()

dur=40
tstart = 40

def get_ac(nn,i=0):
    x = calc_spikes.get_spike_train(nn.mon_spike_e[i],dur,dt,tstart)
    y=numpy.correlate(x,gas,'same')  
    z = numpy.correlate(y,y,'same')

    return z


def get_mean_ac(nn,n=5):
    sh = 0
    z=get_ac(nn,sh+0)
    for i in range(n-1):
        z+=get_ac(nn,sh+1+i)

    plt.plot(z)
    plt.show()

