from brian import ms, mV, pA, second, Hz, siemens, nS
from matplotlib import pyplot, gridspec 
import matplotlib
import numpy
import brian as bb
reload(bb)
import calc_spikes

from avalan import get_avalanches
from periodogram import periodogram
#from line_with_text import MyLine

################################################################################
def get_straightline(figure, sub_e, sub_r, c1x, c1y, c2x, c2y):
    transFigure = figure.transFigure.inverted()
    coord1 = transFigure.transform(sub_e.transData.transform([c1x, c1y]))
    coord2 = transFigure.transform(sub_r.transData.transform([c2x, c2y]))
    line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
        (coord1[1], coord2[1]), linestyle='--',# clip_on=False,
        transform=figure.transFigure, color='gray', alpha=1, zorder=1)
    return line

def plot_pop_raster(net, plot_inh=False):
    """plot rasterplot of whole E/I populations"""
    pyplot.figure()
    bb.raster_plot(net.mon_spike_e)
    if plot_inh:
        pyplot.figure()
        bb.raster_plot(net.mon_spike_i)

def plot_ps_raster(net, chain_n=0, frac = 1., permutate=False, figure=0,
                    dummy_ass=False):
    '''
        plot a raster of the neurons in the phase sequence chain_n
        frac is the fraction of neurons from each assembly to be plotted.
    '''
    a = calc_spikes.get_spike_times_ps(net, chain_n, frac, permutate,
                                        True, dummy_ass)
    if not figure:
        figure = pyplot.figure()
    bb.raster_plot_spiketimes(a)
    pyplot.title('PS %d neuron firing'%chain_n)
    
def plot_pop_fr(net,w=10*ms):
    """plots E/I population FR"""
    pyplot.figure()
    pyplot.subplot(211)
    pyplot.plot(net.mon_rate_e.smooth_rate(w))
    pyplot.xlabel('t [ms]')
    pyplot.ylabel('FR [sp/sec]')
    pyplot.title('FR of E')
    pyplot.subplot(212)
    pyplot.plot(net.mon_rate_i.smooth_rate(w))
    pyplot.xlabel('t [ms]')
    pyplot.ylabel('FR [sp/sec]')
    pyplot.title('FR of I')

def plot_voltage(net, plot_inh=False):
    """plot voltages of measured neurons"""
    pyplot.figure()
    for n in range(len(net.nrn_meas_e)):
        pyplot.subplot(len(net.nrn_meas_e),1,n+1)
        pyplot.plot(net.mon_volt_e.times/ms, 
                            net.mon_volt_e[net.nrn_meas_e[n]]/mV)
        pyplot.ylabel('V [mV]')
        pyplot.xlabel('t [ms]')
        pyplot.legend(['PY %d'%net.nrn_meas_e[n]])

    if plot_inh:
        pyplot.figure()
        for n in range(len(net.nrn_meas_i)):
            pyplot.subplot(len(net.nrn_meas_i),1,n+1)
            pyplot.plot(net.mon_volt_i.times/ms, 
                        net.mon_volt_i[net.nrn_meas_i[n]]/mV)
            pyplot.ylabel('V [mV]')
            pyplot.xlabel('t [ms]')
            pyplot.legend(['IN %d'%net.nrn_meas_i[n]])

def plot_conductance(net, nrn = 0):
    """plot E/I conductances of measured E/I neurons"""
    pyplot.figure()
    for n in range(len(net.nrn_meas_e)):
        pyplot.subplot(len(net.nrn_meas_e),1,n+1)
        pyplot.plot(net.mon_econd_e.times/ms, 
                            net.mon_econd_e[net.nrn_meas_e[n]]/nS)
        pyplot.plot(net.mon_icond_e.times/ms, 
                            net.mon_icond_e[net.nrn_meas_e[n]]/nS)
        pyplot.ylabel('ge,gi,[nS]')
        pyplot.xlabel('t [ms]')
        pyplot.legend(['PY %d'%net.nrn_meas_e[n]])

    pyplot.figure()
    for n in range(len(net.nrn_meas_i)):
        pyplot.subplot(len(net.nrn_meas_i),1,n+1)
        pyplot.plot(net.mon_econd_i.times/ms, 
                            net.mon_econd_i[net.nrn_meas_i[n]]/nS)
        pyplot.plot(net.mon_icond_i.times/ms, 
                            net.mon_icond_i[net.nrn_meas_i[n]]/nS)
        pyplot.ylabel('ge,gi [nS]')
        pyplot.xlabel('t [ms]')
        pyplot.legend(['IN %d'%net.nrn_meas_i[n]])

def plot_currents(net, nrn = 0):
    """plot E/I current of measured E/I neurons"""
    pyplot.figure()
    for n in range(len(net.nrn_meas_e)):
        pyplot.subplot(len(net.nrn_meas_e),1,n+1)
        pyplot.plot(net.mon_ecurr_e.times/ms, 
                            net.mon_ecurr_e[net.nrn_meas_e[n]]/pA)
        pyplot.plot(net.mon_icurr_e.times/ms, 
                            net.mon_icurr_e[net.nrn_meas_e[n]]/pA)
        pyplot.plot(net.mon_icurr_e.times/ms, 
                            (net.mon_ecurr_e[net.nrn_meas_e[n]]+
                            net.mon_icurr_e[net.nrn_meas_e[n]])/pA)
        pyplot.ylabel('Ie,Ii,net [pA]')
        pyplot.xlabel('t [ms]')
        pyplot.legend(['PY %d'%net.nrn_meas_e[n]])

    pyplot.figure()
    for n in range(len(net.nrn_meas_i)):
        pyplot.subplot(len(net.nrn_meas_i),1,n+1)
        pyplot.plot(net.mon_ecurr_i.times/ms, 
                            net.mon_ecurr_i[net.nrn_meas_i[n]]/pA)
        pyplot.plot(net.mon_icurr_i.times/ms, 
                            net.mon_icurr_i[net.nrn_meas_i[n]]/pA)
        pyplot.ylabel('Ie,Ii [pA]')
        pyplot.xlabel('t [ms]')
        pyplot.legend(['IN %d'%net.nrn_meas_i[n]])

def plot_separatrix(net,t_list=[12,13],t_pre=.005,t_post=.025,ps=0):
    '''
        try to plot a separatrix
    '''
    pyplot.figure()
    for t in t_list:
        a,s = calc_spikes.get_alpha_sigma(net,t,t_pre,t_post)
        pyplot.plot(s,a,'-*')
    pyplot.xlabel('$\sigma$ [ms]')
    pyplot.ylabel('$\\alpha$ [# spikes]')

def plot_separatrix_n(nn,t_list=[12,13],t_pre=.005,t_post=.025,ps=0,
                                                    n_spikes=False):
    '''
        try to plot a separatrix
    '''
    reload(calc_spikes)
    figure = pyplot.figure(figsize=(12.,6.))
    gs = gridspec.GridSpec(20,20)
    gs.update(left=.1,wspace=.05,hspace=.05)
    subp = pyplot.subplot(gs[0:11,0:6])
    
    xlim = [-.5,16.5]
    x_ticks = [0,8,16]

    y_ticks = [0,.500,1]
    #y_ticks = [0,100,200,300,400,500,600]
    y_ticks = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
    y_tick_lab = ['0','','','','','50','','','','','100']
    ylim = [0,1.1]

    tas_l = [] # t, alpha, sigma list
    #tas_l.append([17,500,0])
    #tas_l.append([18,500,0])
    #tas_l.append([19,500,0])

    #tms_sig_l = [[0,0],[.200,5],[.400,10],[.600,20],[.800,30]]
    #tms_alpha_l = [[20,400],[21,300],[22,200],[23,100],[24,50]]
    tms_alpha_l = [[17.,1.],[18.,.8],[19.,.6],[20.,.4],[21.,.2],[22.,.1]]
    tms_sig_l = [[0,0],[.200,4],[.400,8],[.600,12],[.800,16]]

    for tms_a in tms_alpha_l:
        for tms_s in tms_sig_l:
            tas_l.append([tms_a[0]+tms_s[0], tms_a[1], tms_s[1]])
    #print tas_l
    for tas in tas_l:
        #reload(calc_spikes)
        a,s = calc_spikes.get_alpha_sigma(nn,tas[0],t_pre,t_post,
                                            n_spikes=n_spikes)
        #print tas,a,s
        #a=a/nn.s_ass
        print a
        a.insert(0,tas[1])
        s.insert(0,tas[2])
        na=numpy.array(a)
        ns=numpy.array(s)
        if a[-1]>.6: color = 'r'
        elif a[-1]>.4: color = 'b'
        else: color= 'g'
        #color= 'g'
        #pyplot.plot(s,a,'-*',color=color)
        pyplot.quiver(ns[:-1],na[:-1],ns[1:]-ns[:-1],na[1:]-na[:-1],
             width=.015, scale_units = 'xy',angles='xy',scale=1.,color=color)
        pyplot.plot(tas[2],tas[1],'o',color=color)
        #pyplot.plot(s[1:],a[1:],'*',color=color)
        pyplot.plot(s[-1],a[-1],'o',color='black')

    #pyplot.plot(numpy.array(tas_l)[:,2],numpy.array(tas_l)[:,1],'o')
    subp.set_xlabel('$\sigma$ [ms]',size=20)
    subp.set_ylabel('$\\alpha$ [% neurons]',size=20)


    subp.set_xticks(x_ticks)
    subp.set_xticklabels(x_ticks,size=20)

    subp.set_yticks(y_ticks)
    subp.set_yticklabels(y_tick_lab,size=20)

    subp.set_xlim(xlim)
    subp.set_ylim(ylim)

    figure.savefig('tfigs/separau_pr'+str(nn.pr_ee)+'pf'+str(nn.pf_ee)+
              'tp'+str(t_pre)+'_'+str(t_post)+'.png', format = 'png')
    #pyplot.show()
    return a,s

def remove_spines(ax,xlim,ylim,xticks,yticks,ylabel,legend,keepx=False):
    for loc, spine in ax.spines.items():
        if loc in ['left']:
            spine.set_position(('outward',0)) # outward by 10 points
        elif loc in ['right','top']: 
            spine.set_color('none') # don't draw spine
        elif loc in ['bottom']: 
            spine.set_position(('outward',10)) # outward by 10 points
            if not  keepx: spine.set_color('none') # don't draw spine
        else:
            raise ValueError('unknown spine location: %s'%loc)

    pyplot.xlim(xlim)
    pyplot.xticks(xticks)
    pyplot.yticks(yticks)
    pyplot.ylim(ylim)
    pyplot.ylabel(ylabel)
    if legend<>[]: pyplot.legend(legend)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def remove_spines_only(ax, keepx=False):
    for loc, spine in ax.spines.items():
        if loc in ['left']:
            spine.set_position(('outward', 0)) # outward by 10 points
        elif loc in ['right','top']: 
            spine.set_color('none') # don't draw spine
        elif loc in ['bottom']: 
            spine.set_position(('outward', 0)) # outward by 10 points
            if not  keepx: spine.set_color('none') # don't draw spine
        else:
            raise ValueError('unknown spine location: %s'%loc)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def plot_rast_V_I_sep(net,nrn=-2,xlim = [16500,17500]):
    """ plots a ratser of a reduced PS,
        conductances, currents, voltage of a neuron in a single figure"""

    tres = .1   # time resolution for conductance
    cxlim = [xlim[0]/tres,xlim[1]/tres] # lim for the vectors(with tres in mind)

    xticks=numpy.arange(xlim[0],xlim[1]+1,200)
    xticks_lab = numpy.arange(0,xlim[1]-xlim[0]+1,200)

    figure = pyplot.figure(figsize=(12.,8.))
    gs = gridspec.GridSpec(20,12)
    gs.update(left=.1,wspace=.0,hspace=.0)

    ylab_xposit = xlim[0]-130

    ################################################################
    #sub_rast = pyplot.subplot(gs[0:6,0:12])
    sub_rast = pyplot.subplot(gs[14:20,1:12])
    m = bb.Monitor()
    m.source = []
    m.spikes = calc_spikes.get_spike_times_ps(net,0,.1)
    bb.raster_plot(m,color=(0,0,0))#, #showgrouplines=True,
                    #spacebetweengroups=0.1, grouplinecol=(0.5, 0.5, 0.5))

    # spikes of measured neuron in red o.,
    # now it works automatically just with neurons from the last group

    nrn_spikes_all = net.mon_spike_e[net.nrn_meas_e[nrn]]
    nrn_spikes = nrn_spikes_all[numpy.logical_and(
            nrn_spikes_all >= xlim[0]/1000., nrn_spikes_all <=xlim[1]/1000.)]

    raster_sp_x = nrn_spikes*1000.
    raster_sp_y = (491+nrn)*numpy.ones(len(nrn_spikes))

    pyplot.plot(raster_sp_x,raster_sp_y,'o',color='red')
    pyplot.plot(raster_sp_x,raster_sp_y,'.',color='red')

    sub_rast.set_xlim(xlim)
    sub_rast.set_xticks(xticks)
    sub_rast.set_xticklabels([])
    sub_rast.set_xlabel('')

    # put some lines between groups
    for i in range(net.n_ass):
        sub_rast.plot(numpy.arange(xlim[0],xlim[1]),
                                    (i+1)*50.*numpy.ones(1000),'gray')

    # put an arrow for the neuron we record from
    #pyplot.text(xlim[1]+2,490+nrn,'$\leftarrow$',size=18)
    #pyplot.text(xlim[0]+500-18,-60,'$\lightning$',size=25)
    pyplot.text(xlim[0]+500-18,-60,'$\Uparrow$',size=25)
    #pyplot.text(xlim[0]+500+23,-80,'Stimulation',size=18)

    yticks=[0,100,200,300,400,500]
    yticks=[0,250,500]
    sub_rast.set_ylabel('',size=20)
    sub_rast.set_yticks(yticks)
    sub_rast.set_yticklabels(yticks,size=20)
    sub_rast.xaxis.set_tick_params(width=1,length=5)
    sub_rast.yaxis.set_tick_params(width=1,length=5)

    #sub_rast.set_ylabel('Neurons',size=18)
    sub_rast.text(ylab_xposit,375,'Neuron #',rotation = 'vertical',size=20)

    sub_rast.set_xticklabels(xticks_lab, size=20)
    pyplot.xlabel('Time [ms]',size=20)

    sub_rast.text(xticks[0]-200,500.,'C',size=28,color='black')


    ################################################################
    #sub_curr = pyplot.subplot(gs[9:13,0:12])
    sub_curr = pyplot.subplot(gs[0:5,1:12])
    yticks=[-200,0,200,400]
    curr_e= net.mon_ecurr_e[net.nrn_meas_e[nrn]]/pA+200
    curr_i= net.mon_icurr_e[net.nrn_meas_e[nrn]]/pA
    volt_e= net.mon_volt_e[net.nrn_meas_e[nrn]]/mV # measured voltage here
    curr_leak = 10.*nS*(-60.-volt_e)*mV/pA
    curr_maxe=(curr_e[cxlim[0]:cxlim[1]]).max()
    curr_maxi=(curr_i[cxlim[0]:cxlim[1]]).max()
    curr_mine=(curr_e[cxlim[0]:cxlim[1]]).min()
    curr_mini=(curr_i[cxlim[0]:cxlim[1]]).min()
    pyplot.plot(net.mon_ecurr_e.times/ms, curr_e,'r')
    pyplot.plot(net.mon_ecurr_e.times/ms, curr_i,'b')
    pyplot.plot(net.mon_ecurr_e.times/ms, curr_e+curr_i+curr_leak,'black')
    ylim=[1.2*min(curr_mine,curr_mini),1.2*max(curr_maxe,curr_maxi)]
    remove_spines(sub_curr,xlim=xlim,ylim=ylim,xticks=[],yticks=yticks,
                    #ylabel='I [pA]',legend= ['$I_{exc}$','$I_{inh}$'])
                    ylabel='I [pA]',legend= [])
    sub_curr.set_ylabel('',size=20)
    sub_curr.text(ylab_xposit,370,'Current [pA]',rotation = 'vertical',size=20)
    sub_curr.set_yticks(yticks)
    sub_curr.set_yticklabels(yticks,size=20)
    sub_curr.yaxis.set_tick_params(width=1,length=5)

    #sub_curr.text(xticks[-1]-00,-200.,'$I_{inh}$',color='red',size=18)
    #sub_curr.text(xticks[-1]-00,50.,'$I_{tot}$',color='black',size=18)
    #sub_curr.text(xticks[-1]-00,250.,'$I_{exc}$',color='green',size=18)
    sub_curr.text(xticks[-2]-5,-320.,'Inhibition',size=20,color='b',zorder=3)
    sub_curr.text(xticks[-2]-5,80.,'Total',size=20,zorder=3)
    sub_curr.text(xticks[-2]-5,300.,'Excitation',size=20,color='r',zorder=3)

    sub_curr.text(xticks[0]-200,500.,'A',size=28,color='black')

    ################################################################
    #raster_sp_x = nrn_spikes*1000 
    #raster_sp_y = (491+nrn)*numpy.ones(len(nrn_spikes))

    #sub_volt = pyplot.subplot(gs[16:20,0:12])
    sub_volt = pyplot.subplot(gs[7:12,1:12])
    for i,v in enumerate(volt_e):
        if v<-59 and volt_e[i-1]>-55:
            #pyplot.vlines((i-1)*tres,-50,-40)
            volt_e[i-1] = -40
            #volt_e[i] = -40
            #volt_e[i+1] = -40
            #print i
    pyplot.plot(net.mon_volt_e.times/ms, volt_e,'black')

    volt_sp_x = nrn_spikes*1000 
    volt_sp_y = -40*numpy.ones(len(volt_sp_x)) 
    
    pyplot.plot(volt_sp_x,volt_sp_y,'o',color='red')
    
    #for i,v in enumerate(volt_e):
        #if volt_e[i-1]>-45:
            #pyplot.plot((i-1)*tres,-40,'o',color='red')

    ylim=[-62,-39]
    yticks=[-60,-50,-40]
    #remove_spines(sub_volt,xlim=xlim,ylim=ylim,
                #xticks=xticks,yticks=yticks,
                #ylabel='V [mV]',legend= [],keepx=True)
    remove_spines(sub_volt,xlim=xlim,ylim=ylim,
                xticks=[],yticks=yticks,
                ylabel='V [mV]',legend= [])
    #sub_volt.set_xticklabels(xticks_lab, size=18)
    #pyplot.xlabel('Time [ms]',size=18)
    sub_volt.set_ylabel('',size=20)
    sub_volt.text(ylab_xposit,-40,'Voltage [mV]',rotation = 'vertical',size=20)
    sub_volt.set_yticks(yticks)
    sub_volt.set_yticklabels(yticks,size=20)
    sub_volt.yaxis.set_tick_params(width=1,length=5)

    sub_volt.plot(numpy.arange(xlim[0],xlim[1]),
                                -50.*numpy.ones(1000),'-.',color = 'gray')
    
    sub_volt.text(xticks[0]-200,-40.,'B',size=28,color='black')
    #sub_volt.text(xticks[-1]-00,-55.,'$V_{m}$',color='black',size=18)

    # draw vertical lines between spikes in raster and voltage trace

    linii_nekvi=[]
    curr_sp_y = 400.
    for i in range(len(volt_sp_x)):
        #linii_nekvi.append(get_straightline(figure,sub_rast,sub_volt,
        #        raster_sp_x[i],raster_sp_y[i],volt_sp_x[i],volt_sp_y[i]))
        linii_nekvi.append(get_straightline(figure,sub_rast,sub_curr,
                raster_sp_x[i],raster_sp_y[i],volt_sp_x[i],curr_sp_y))
    '''
    transFigure = figure.transFigure.inverted()
    c1x = raster_sp_x[0]
    c1y = raster_sp_y[0]
    c2x = volt_sp_x[0]
    c2y = volt_sp_y[0]
    coord1 = transFigure.transform(sub_rast.transData.transform([c1x,c1y]))
    coord2 = transFigure.transform(sub_volt.transData.transform([c2x,c2y]))
    line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                        transform=figure.transFigure,color='gray',alpha=.9)

    print c1x,c1y,c2x,c2y
    print coord1, coord2
    '''
    figure.lines = linii_nekvi

    loc='tfigs/rcv/'
    prefix = 'rccv_'
    fname = 'prf'+str(net.pr_ee) + str(net.pf_ee)+str(-nrn)+'_int'+str(xlim[0])
    figure.savefig(loc+prefix+ fname +'.eps', format = 'eps')
    figure.savefig(loc+prefix+ fname +'.png', format = 'png')
    figure.savefig(loc+prefix+ fname +'.pdf', format = 'pdf')

    pyplot.show()


def plot_rast_V_I_disc(net, nrn=-2, xlim0=[19375,19625],
            xlim1 = [25625,26375], xlim2=[38625,36375], disc_only=False):
    """
        For plotting Fig 2
        - plots currents, voltage of a neuron in a single figure
        - plots a ratser of a PS with a discontinious x axis for
        showing off the dummpy plasticity
        
        xlimX : gives the limits of the x axes for the different plots
            X=0 : limits of pre-pff-boost
            X=1 : limits of discrete sequence sfter boost
            X=1 : limits of continuous sequence after boost

        puts an extra raster of a continuous sequence replay
    """
    from figure_subplots import draw_evoked_rasta2
    tres = .1   # time resolution for conductance
    subpzorder = 2

    figure = pyplot.figure(figsize=(18., 12.))
    grid_y = 26
    if disc_only:
        grid_y = 18
    gs = gridspec.GridSpec(grid_y, 13)
    gs.update(left=.04, wspace=.15, hspace=.0)

    ylab_xposit = xlim0[0] - 80

    # label size
    labsize = 20
    ################################################################
    dx_tick = 50
    xticks0 = numpy.arange(xlim0[0], xlim0[1]+1, dx_tick)
    xticks0_lab = numpy.arange(0, xlim0[1]-xlim0[0]+1, dx_tick)

    # first tick in second plot in plotting time
    shift_xtick = dx_tick - (xlim0[1]-xlim0[0]) % dx_tick 
    # last tick
    first_xtick= ((xlim0[1]-xlim0[0]+dx_tick)//dx_tick)*dx_tick
    last_xtick = first_xtick + xlim1[1]-xlim1[0] + 1
    xticks1 = numpy.arange(xlim1[0] + shift_xtick, xlim1[1]+1, dx_tick)
    xticks1_lab = numpy.arange(first_xtick, last_xtick, dx_tick)

    #xticks=numpy.arange(xlim[0], xlim[1]+1, 200)
    #xticks_lab = numpy.arange(0, xlim[1]-xlim[0]+1, 200)

    ################################################################
    sub_rast0 = pyplot.subplot(gs[12:18, 1:4], zorder = subpzorder )
    m = bb.Monitor()
    m.source = []
    frac = .1
    m.spikes = calc_spikes.get_spike_times_ps(net, 0, frac)
    bb.raster_plot(m, color=(0, 0, 0))
    #sub_rast0.plot(numpy.array(m.spikes)[:,0],numpy.array(m.spikes)[:,1], '.')

    # spikes of measured neuron in red o.,
    # now it works only with neurons from the last group
    nrn_spikes_all = net.mon_spike_e[net.nrn_meas_e[nrn]]
    nrn_spikes0 = nrn_spikes_all[numpy.logical_and(
        nrn_spikes_all >= xlim0[0]/1000., nrn_spikes_all <= xlim0[1]/1000.)]
    nrn_spikes1 = nrn_spikes_all[numpy.logical_and(
        nrn_spikes_all >= xlim1[0]/1000., nrn_spikes_all <= xlim1[1]/1000.)]

    raster_sp_x0 = nrn_spikes0*1000.
    raster_sp_y0 = (net.nrn_meas_e[nrn])*numpy.ones(len(nrn_spikes0))*frac

    pyplot.plot(raster_sp_x0, raster_sp_y0, 'o', color='red', zorder=3)
    pyplot.plot(raster_sp_x0, raster_sp_y0, '.', color='red', zorder=3)

    sub_rast0.set_xlim(xlim0)
    #sub_rast0.set_xticks([])
    #sub_rast0.set_xticklabels([])
    sub_rast0.set_xlabel('')

    # put some lines between groups
    for i in range(net.n_ass):
        sub_rast0.plot(numpy.arange(xlim0[0], xlim0[1]),
                    (i+1)*frac*net.s_ass*numpy.ones(xlim0[1]-xlim0[0]),
                    'gray', zorder=1)

    yticks=[0, 2500, 5000]
    yticks=[0, 250, 500]
    sub_rast0.set_ylabel('', size=20)
    sub_rast0.set_yticks(yticks)
    sub_rast0.set_yticklabels(yticks, size=labsize)
    sub_rast0.xaxis.set_tick_params(width=1, length=5)
    sub_rast0.yaxis.set_tick_params(width=1, length=5)

    sub_rast0.set_xlim(xlim0)
    sub_rast0.set_xticks(xticks0)
    sub_rast0.set_xticks([])
    sub_rast0.set_xticklabels(xticks0_lab, size=20)
    sub_rast0.set_xticklabels([], size=20)
    sub_rast0.text(xticks0[0]-130, yticks[-1], 'C', size=28, color='black')
    sub_rast0.text(ylab_xposit, 375, 'Neuron #', rotation = 'vertical',
                    size=20)
    pyplot.text(xlim0[0]+125-12, -60, '$\Uparrow$', size=25)

    sub_rast0.yaxis.tick_right()
    sub_rast0.spines['right'].set_visible(False)

    ###
    sub_rast = pyplot.subplot(gs[12:18, 4:13], zorder = subpzorder)
    m = bb.Monitor()
    m.source = []
    m.spikes = calc_spikes.get_spike_times_ps(net, 0, frac)
    bb.raster_plot(m, color=(0, 0, 0))
    #pyplot.plot(numpy.array(m.spikes)[:,0],numpy.array(m.spikes)[:,1], '.')

    raster_sp_x1 = nrn_spikes1*1000.
    raster_sp_y1 = (net.nrn_meas_e[nrn])*numpy.ones(len(nrn_spikes1))*frac

    pyplot.plot(raster_sp_x1, raster_sp_y1, 'o', color='red')
    pyplot.plot(raster_sp_x1, raster_sp_y1, '.', color='red')

    # put some lines between groups
    for i in range(net.n_ass):
        sub_rast.plot(numpy.arange(xlim1[0], xlim1[1]),
                    (i+1)*frac*net.s_ass*numpy.ones(xlim1[1]-xlim1[0]), 'gray')

    sub_rast.set_xlim(xlim1)
    sub_rast.set_xticks(xticks1)
    sub_rast.set_xticks([])
    sub_rast.set_xlabel('')
    sub_rast.spines['left'].set_visible(False)

    sub_rast.set_ylabel('', size=20)
    sub_rast.set_yticks([])
    sub_rast.xaxis.set_tick_params(width=1, length=5)
    sub_rast.yaxis.set_tick_params(width=1, length=5)
    sub_rast.set_xticklabels([], size=20)

    pyplot.text(xlim1[0]+375-12,-60, '$\Uparrow$', size=25)
    #sub_rast.set_xticklabels(xticks1_lab, size=20)
    #pyplot.xlabel('Time [ms]', size=20)

    # size of the diagonal lines in axes coordinates
    dx0 = .012 
    dy0 = .04 
    # the second subplot has different size than the prev
    dx1 = dx0*3./8 
    dy1 = dy0*1. 

    kwargs = dict(transform=sub_rast0.transAxes, color='k', clip_on=False)
    sub_rast0.yaxis.tick_left()
    # top-right diagonal
    sub_rast0.plot((1-dx0, 1+dx0), (1-dy0, 1+dy0), **kwargs)
    # bottom-right diagonal
    sub_rast0.plot((1-dx0, 1+dx0), (-dy0, +dy0), **kwargs)

    kwargs.update(transform=sub_rast.transAxes)  # switch to the bottom axes
    # top-left diagonal
    sub_rast.plot((-dx1, +dx1), (1-dy1, 1+dy1), **kwargs)
    # bottom-left diagonal
    sub_rast.plot((-dx1, dx1), (-dy1, +dy1), **kwargs)

    ################################################################
    sub_curr0 = pyplot.subplot(gs[0:5, 1:4], zorder = subpzorder)
    sub_curr0.patch.set_facecolor('None')

    #lim for the vectors(with tres in mind)
    cx0lim = [xlim0[0]/tres, xlim0[1]/tres]
    cx1lim = [xlim1[0]/tres, xlim1[1]/tres]

    yticks=[-200, 0, 200, 400]
    curr_e= net.mon_ecurr_e[net.nrn_meas_e[nrn]]/pA+200
    curr_i= net.mon_icurr_e[net.nrn_meas_e[nrn]]/pA
    volt_e= net.mon_volt_e[net.nrn_meas_e[nrn]]/mV # measured voltage here
    curr_leak = 10.*nS*(-60.-volt_e)*mV/pA
    curr_maxe = max((curr_e[cx0lim[0]:cx0lim[1]]).max(),
                    (curr_e[cx1lim[0]:cx1lim[1]]).max())
    curr_maxi = max((curr_i[cx0lim[0]:cx0lim[1]]).max(),
                    (curr_i[cx1lim[0]:cx1lim[1]]).max())
    curr_mine = min((curr_e[cx0lim[0]:cx0lim[1]]).min(),
                    (curr_e[cx1lim[0]:cx1lim[1]]).min())
    curr_mini = min((curr_i[cx0lim[0]:cx0lim[1]]).min(),
                    (curr_i[cx1lim[0]:cx1lim[1]]).min())

    sub_curr0.plot(net.mon_ecurr_e.times/ms, curr_e, 'r', zorder=3)
    sub_curr0.plot(net.mon_ecurr_e.times/ms, curr_i, 'b', zorder=1)
    sub_curr0.plot(net.mon_ecurr_e.times/ms, curr_e+curr_i+curr_leak,
                    'black', zorder=2)

    ylim=[1.02*min(curr_mine, curr_mini), 1.1*max(curr_maxe, curr_maxi)]
    remove_spines(sub_curr0, xlim=xlim0, ylim=ylim, xticks=[], yticks=yticks,
                    ylabel='I [pA]', legend= [])
    sub_curr0.set_ylabel('', size=20)
    sub_curr0.text(ylab_xposit, 330, 'Current [pA]',
                    rotation = 'vertical', size=20)
    sub_curr0.text(xticks0[0]-130, 500., 'A', size=28, color='black')
    sub_curr0.set_yticks(yticks)
    sub_curr0.set_yticklabels(yticks, size=20)
    sub_curr0.yaxis.set_tick_params(width=1, length=5)


    sub_curr = pyplot.subplot(gs[0:5, 4:13], zorder = subpzorder)
    sub_curr.patch.set_facecolor('None')

    pyplot.plot(net.mon_ecurr_e.times/ms, curr_e, 'r')
    pyplot.plot(net.mon_ecurr_e.times/ms, curr_i, 'b')
    pyplot.plot(net.mon_ecurr_e.times/ms, curr_e+curr_i+curr_leak, 'black')

    ylim=[1.2*min(curr_mine, curr_mini), 1.2*max(curr_maxe, curr_maxi)]
    remove_spines(sub_curr, xlim=xlim1, ylim=ylim, xticks=[], yticks=[],
                    ylabel='', legend= [])
    sub_curr.set_ylabel('', size=20)
    sub_curr.yaxis.set_tick_params(width=1, length=5)


    sub_curr.text(xlim1[1]-87, -320., 'Inhibition', size=20,
                                        color='b', zorder=3)
    sub_curr.text(xlim1[1]-45, 80., 'Total', size=20, zorder=3)
    sub_curr.text(xlim1[1]-92, 300., 'Excitation', size=20,
                                        color='r', zorder=3)
    '''
    '''
    sub_curr0.spines['right'].set_visible(False)
    sub_curr.spines['left'].set_visible(False)

    sub_curr0.yaxis.tick_left()
    '''
    kwargs.update(transform=sub_curr0.transAxes)  # switch to the bottom axes
    sub_rast0.yaxis.tick_left()
    # top-right diagonal
    sub_rast0.plot((1-dx0, 1+dx0), (1-dy0, 1+dy0), **kwargs)
    # bottom-right diagonal
    sub_rast0.plot((1-dx0, 1+dx0), (-dy0, +dy0), **kwargs)

    kwargs.update(transform=sub_curr.transAxes)  # switch to the bottom axes
    # top-left diagonal
    sub_rast.plot((-dx1, +dx1), (1-dy1, 1+dy1), **kwargs)
    # bottom-left diagonal
    sub_rast.plot((-dx1, dx1), (-dy1, +dy1), **kwargs)
    '''

    ################################################################
    #raster_sp_x = nrn_spikes*1000 
    #raster_sp_y = (491+nrn)*numpy.ones(len(nrn_spikes))
    #sub_volt = pyplot.subplot(gs[16:20,0:12])
    sub_volt0 = pyplot.subplot(gs[6:11,1:4], zorder = subpzorder)
    sub_volt0.patch.set_facecolor('None')
    ylim=[-61,-39]
    yticks=[-60,-50,-40]

    for i,v in enumerate(volt_e):
        if v<-59 and volt_e[i-1]>-55:
            volt_e[i-1] = -40

    volt_sp_x0 = nrn_spikes0*1000 
    volt_sp_y0 = -40*numpy.ones(len(volt_sp_x0)) 
    
    pyplot.plot(net.mon_volt_e.times/ms, volt_e,'black')
    pyplot.plot(volt_sp_x0, volt_sp_y0, 'o', color='red')

    remove_spines(sub_volt0, xlim=xlim0, ylim=ylim,
                xticks=[], yticks=yticks,
                ylabel='V [mV]', legend= [])
    sub_volt0.set_ylabel('', size=20)
    sub_volt0.text(ylab_xposit, -41, 'Voltage [mV]',
                    rotation='vertical', size=20)
    sub_volt0.set_yticks(yticks)
    sub_volt0.set_yticklabels(yticks, size=20)
    sub_volt0.yaxis.set_tick_params(width=1, length=5)
    sub_volt0.plot(numpy.arange(xlim0[0],xlim0[1]),
            -50.*numpy.ones(xlim0[1]-xlim0[0]), '-.', color='gray')
    sub_volt0.text(xticks0[0]-130, -40., 'B', size=28, color='black')


    sub_volt = pyplot.subplot(gs[6:11,4:13], zorder = subpzorder)
    sub_volt.patch.set_facecolor('None')

    volt_sp_x1 = nrn_spikes1*1000 
    volt_sp_y1 = -40*numpy.ones(len(volt_sp_x1)) 
    
    pyplot.plot(net.mon_volt_e.times/ms, volt_e,'black')
    pyplot.plot(volt_sp_x1, volt_sp_y1, 'o', color='red')
 
    remove_spines(sub_volt, xlim=xlim1, ylim=ylim,
                xticks=[], yticks=yticks,
                ylabel='', legend= [])
    sub_volt.set_yticks([])
    sub_volt.plot(numpy.arange(xlim1[0],xlim1[1]),
            -50.*numpy.ones(xlim1[1]-xlim1[0]), '-.', color='gray')

    sub_volt0.spines['right'].set_visible(False)
    sub_volt.spines['left'].set_visible(False)

    # draw vertical lines between spikes in raster and voltage trace
    linii_nekvi=[]
    curr_sp_y = 400.
    for i in range(len(volt_sp_x0)):
        linii_nekvi.append(get_straightline(figure, sub_rast0, sub_curr0,
                raster_sp_x0[i], raster_sp_y0[i], volt_sp_x0[i], curr_sp_y))

    for i in range(len(volt_sp_x1)):
        linii_nekvi.append(get_straightline(figure, sub_rast, sub_curr,
                raster_sp_x1[i], raster_sp_y1[i], volt_sp_x1[i], curr_sp_y))

    if disc_only:
        bar_x = xlim1[-1] - 100
        bar_y = -60
        x_len = 100
        sub_volt.text(bar_x+.15*x_len, bar_y+2, str(x_len)+' ms', size=20)
        transFigure = figure.transFigure.inverted()
        coord1 = transFigure.transform(
                        sub_volt.transData.transform([bar_x+0, bar_y]))
        coord2 = transFigure.transform(
                        sub_volt.transData.transform([bar_x+0+x_len, bar_y]))
        line_bar = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                        (coord1[1], coord2[1]), transform=figure.transFigure, 
                        color='black', linewidth=3.)

        linii_nekvi.append(line_bar)
        figure.lines = linii_nekvi
        from random import randint
        loc='tfigs/rcv8/'
        prefix = 'rccv_'
        fname = 'prf' + str(net.pr_ee) + str(net.pf_ee_new) + '_' + \
            str(-nrn) + '_int' + str(xlim1[0]) + '_' + str(randint(0, 99999))
        figure.savefig(loc+prefix+fname+'rand_.eps', format='eps')
        figure.savefig(loc+prefix+fname+'rand_.png', format='png')
        figure.savefig(loc+prefix+fname+'rand_.pdf', format='pdf')
        pyplot.show()
        return 0


    ################################################################
    # raster for continious ASS at the bottom of the figure

    fname = 'contASS_pr0.1pfboost0.04frac0.1.npz'
    #fname = 'contin_modified.npz'
    xlim0 = [19375, 19625]
    xlim2 = [28625, 29375]

    #xticks=numpy.arange(x0, x1+1, 200)
    #xticks_lab = numpy.arange(0, x1-x0+1, 200)
    xticks=[]

    # first tick in second plot in plotting time
    shift_xtick = dx_tick - (xlim0[1]-xlim0[0]) % dx_tick 
    # last tick
    first_xtick= ((xlim0[1]-xlim0[0]+dx_tick)//dx_tick)*dx_tick
    last_xtick = first_xtick + xlim2[1]-xlim2[0] + 1
    xticks2 = numpy.arange(xlim2[0] + shift_xtick, xlim2[1]+1, dx_tick)
 
    sub_rast20 = pyplot.subplot(gs[20:26, 1:4], zorder = subpzorder)
    draw_evoked_rasta2(fname, '', x0=xlim0[0], x1=xlim0[1])
    yticks=[0, 100, 200, 300, 400, 500]
    yticks=[0, 250, 500]
    #yticks=[0, 2500, 5000]
    sub_rast20.set_xticks(xticks0)
    sub_rast20.set_xticks([])
    sub_rast20.set_xticklabels([])
    sub_rast20.set_yticks(yticks)
    sub_rast20.set_yticklabels(yticks, size=20)
    sub_rast20.text(xlim0[0]-80, 375, 'Neuron #',
                    rotation = 'vertical', size=20)
    sub_rast20.text(xlim0[0]-130, yticks[-1], 'D', size=28, color='black')
    sub_rast20.spines['right'].set_visible(False)
    sub_rast20.xaxis.set_tick_params(width=1, length=5)
    sub_rast20.yaxis.set_tick_params(width=1, length=5)
    pyplot.text(xlim0[0]+125-12,-60, '$\Uparrow$', size=25)

    kwargs = dict(transform=sub_rast20.transAxes, color='k', clip_on=False)
    sub_rast20.yaxis.tick_left()
    # top-right diagonal
    sub_rast20.plot((1-dx0, 1+dx0), (1-dy0, 1+dy0), **kwargs)
    # bottom-right diagonal
    sub_rast20.plot((1-dx0, 1+dx0), (-dy0, +dy0), **kwargs)

    #####
    sub_rast2 = pyplot.subplot(gs[20:26, 4:13], zorder = subpzorder)
    draw_evoked_rasta2(fname, '', x0=xlim2[0], x1=xlim2[1])

    sub_rast2.xaxis.set_tick_params(width=1, length=5)
    sub_rast2.yaxis.set_tick_params(width=1, length=5)
    #sub_rast.set_ylabel('Neurons',size=18)
    sub_rast2.set_xticks(xticks2)
    sub_rast2.set_xticks([])
    sub_rast2.set_xticklabels([])
    #pyplot.xlabel('Time [ms]', size=20)

    sub_rast2.spines['left'].set_visible(False)

    kwargs.update(transform=sub_rast2.transAxes)  # switch to the bottom axes
    # top-left diagonal
    sub_rast2.plot((-dx1, +dx1), (1-dy1, 1+dy1), **kwargs)
    # bottom-left diagonal
    sub_rast2.plot((-dx1, dx1), (-dy1, +dy1), **kwargs)
    pyplot.text(xlim2[0]+375-12,-60, '$\Uparrow$', size=25)

    '''
    bar_x = xlim2[-1] - 100
    bar_y = -100
    x_len = 100
    pyplot.text(bar_x+.15*x_len, bar_y+20, str(x_len)+' ms', size=20)
    transFigure = figure.transFigure.inverted()
    coord1 = transFigure.transform(
                    sub_rast2.transData.transform([bar_x+0, bar_y]))
    coord2 = transFigure.transform(
                    sub_rast2.transData.transform([bar_x+0+x_len, bar_y]))
    '''
    bar_x = xlim1[-1] - 100
    bar_y = -60
    x_len = 100
    sub_volt.text(bar_x+.15*x_len, bar_y+2, str(x_len)+' ms', size=20)
    transFigure = figure.transFigure.inverted()
    coord1 = transFigure.transform(
                    sub_volt.transData.transform([bar_x+0, bar_y]))
    coord2 = transFigure.transform(
                    sub_volt.transData.transform([bar_x+0+x_len, bar_y]))

    line_bar = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                        (coord1[1], coord2[1]), transform=figure.transFigure, 
                        color='black', linewidth=3.)

    linii_nekvi.append(line_bar)
    figure.lines = linii_nekvi

    #1/0
    #pyplot.show()
    #return 0

    from random import randint
    loc='tfigs/rcv7/'
    prefix = 'rccv_'
    fname = 'prf' + str(net.pr_ee) + str(net.pf_ee) + '_' + \
        str(-nrn) + '_int' + str(xlim1[0]) + '_' + str(randint(0, 99999))
    figure.savefig(loc+prefix+fname+'.eps', format='eps')
    figure.savefig(loc+prefix+fname+'.png', format='png')
    figure.savefig(loc+prefix+fname+'.pdf', format='pdf')

    #pyplot.close()

def plot_rast_V_I_contraster(net, nrn=-2, xlim = [16500,17500]):
    """ 
        For plotting Fig 2
        plots a ratser of a reduced PS,
        conductances, currents, voltage of a neuron in a single figure

        puts an extra raster of a continuous sequence replay
        
    """
    from figure_subplots import draw_evoked_rasta2
    tres = .1   # time resolution for conductance

    figure = pyplot.figure(figsize=(18., 12.))
    gs = gridspec.GridSpec(25, 12)
    gs.update(left=.04, wspace=.0, hspace=.0)
    ylab_xposit = xlim[0] - 80

    ################################################################
    # raster for continious ASS at the bottom of the figure
    x0 = 20500
    x1 = 21500

    xticks=numpy.arange(x0, x1+1, 200)
    xticks_lab = numpy.arange(0, x1-x0+1, 200)

    sub_rast2 = pyplot.subplot(gs[19:25, 1:12])

    fname = 'contin_ass_pf0.06pr0.06.npz'
    draw_evoked_rasta2(fname, '', x0=x0, x1=x1)

    yticks=[0, 100, 200, 300, 400, 500]
    yticks=[0, 2500, 5000]
    sub_rast2.set_yticks(yticks)
    sub_rast2.set_yticklabels(yticks, size=20)
    sub_rast2.xaxis.set_tick_params(width=1, length=5)
    sub_rast2.yaxis.set_tick_params(width=1, length=5)
    #sub_rast.set_ylabel('Neurons',size=18)
    sub_rast2.text(x0-80, 3750, 'Neuron #', rotation = 'vertical', size=20)
    sub_rast2.set_xticks(xticks)
    sub_rast2.set_xticklabels(xticks_lab, size=20)
    pyplot.xlabel('Time [ms]', size=20)
    sub_rast2.text(xticks[0]-130, 5000., 'D', size=28, color='black')


    ################################################################
    xticks=numpy.arange(xlim[0], xlim[1]+1, 200)
    xticks_lab = numpy.arange(0, xlim[1]-xlim[0]+1, 200)
    cxlim = [xlim[0]/tres, xlim[1]/tres] #lim for the vectors(with tres in mind)

    ################################################################
    sub_rast = pyplot.subplot(gs[12:18, 1:12])
    m = bb.Monitor()
    m.source = []
    frac = 1.
    m.spikes = calc_spikes.get_spike_times_ps(net, 0, frac)
    bb.raster_plot(m, color=(0, 0, 0))#, #showgrouplines=True,
                    #spacebetweengroups=0.1, grouplinecol=(0.5, 0.5, 0.5))

    # spikes of measured neuron in red o.,
    # now it works automatically just with neurons from the last group
    nrn_spikes_all = net.mon_spike_e[net.nrn_meas_e[nrn]]
    nrn_spikes = nrn_spikes_all[numpy.logical_and(
            nrn_spikes_all >= xlim[0]/1000., nrn_spikes_all <= xlim[1]/1000.)]

    raster_sp_x = nrn_spikes*1000.
    raster_sp_y = (net.nrn_meas_e[nrn])*numpy.ones(len(nrn_spikes))

    pyplot.plot(raster_sp_x, raster_sp_y, 'o', color='red')
    pyplot.plot(raster_sp_x, raster_sp_y, '.', color='red')

    sub_rast.set_xlim(xlim)
    sub_rast.set_xticks(xticks)
    sub_rast.set_xticklabels([])
    sub_rast.set_xlabel('')

    # put some lines between groups
    for i in range(net.n_ass):
        sub_rast.plot(numpy.arange(xlim[0],xlim[1]),
                        (i+1)*frac*net.s_ass*numpy.ones(1000),'gray')

    # put an arrow for the neuron we record from
    #pyplot.text(xlim[0]+500-18, -60, '$\Uparrow$', size=25)
    yticks=[0, 100, 200, 300, 400, 500]
    yticks=[0, 2500, 5000]
    sub_rast.set_ylabel('', size=20)
    sub_rast.set_yticks(yticks)
    sub_rast.set_yticklabels(yticks, size=20)
    sub_rast.xaxis.set_tick_params(width=1, length=5)
    sub_rast.yaxis.set_tick_params(width=1, length=5)
    sub_rast.text(ylab_xposit, 3750, 'Neuron #', rotation = 'vertical', size=20)
    #sub_rast.set_xticklabels(xticks_lab, size=20)
    #pyplot.xlabel('Time [ms]', size=20)
    sub_rast.text(xticks[0]-130, 5000., 'C', size=28, color='black')

    ################################################################
    #sub_curr = pyplot.subplot(gs[9:13,0:12])
    sub_curr = pyplot.subplot(gs[0:5,1:12])
    yticks=[-200,0,200,400]
    curr_e= net.mon_ecurr_e[net.nrn_meas_e[nrn]]/pA+200
    curr_i= net.mon_icurr_e[net.nrn_meas_e[nrn]]/pA
    volt_e= net.mon_volt_e[net.nrn_meas_e[nrn]]/mV # measured voltage here
    curr_leak = 10.*nS*(-60.-volt_e)*mV/pA
    curr_maxe=(curr_e[cxlim[0]:cxlim[1]]).max()
    curr_maxi=(curr_i[cxlim[0]:cxlim[1]]).max()
    curr_mine=(curr_e[cxlim[0]:cxlim[1]]).min()
    curr_mini=(curr_i[cxlim[0]:cxlim[1]]).min()
    pyplot.plot(net.mon_ecurr_e.times/ms, curr_e,'r')
    pyplot.plot(net.mon_ecurr_e.times/ms, curr_i,'b')
    pyplot.plot(net.mon_ecurr_e.times/ms, curr_e+curr_i+curr_leak,'black')
    ylim=[1.2*min(curr_mine,curr_mini),1.2*max(curr_maxe,curr_maxi)]
    remove_spines(sub_curr,xlim=xlim,ylim=ylim,xticks=[],yticks=yticks,
                    #ylabel='I [pA]',legend= ['$I_{exc}$','$I_{inh}$'])
                    ylabel='I [pA]',legend= [])
    sub_curr.set_ylabel('',size=20)
    sub_curr.text(ylab_xposit, 370, 'Current [pA]',
                    rotation = 'vertical', size=20)
    sub_curr.set_yticks(yticks)
    sub_curr.set_yticklabels(yticks,size=20)
    sub_curr.yaxis.set_tick_params(width=1, length=5)
    #sub_curr.text(xticks[-1]-00,-200.,'$I_{inh}$',color='red',size=18)
    #sub_curr.text(xticks[-1]-00,50.,'$I_{tot}$',color='black',size=18)
    #sub_curr.text(xticks[-1]-00,250.,'$I_{exc}$',color='green',size=18)
    sub_curr.text(xticks[-2]-5, -320., 'Inhibition', size=20,
                                        color='b', zorder=3)
    sub_curr.text(xticks[-2]-5, 80., 'Total', size=20, zorder=3)
    sub_curr.text(xticks[-2]-5, 300., 'Excitation', size=20,
                                        color='r', zorder=3)
    sub_curr.text(xticks[0]-130, 500., 'A', size=28, color='black')

    ################################################################
    #raster_sp_x = nrn_spikes*1000 
    #raster_sp_y = (491+nrn)*numpy.ones(len(nrn_spikes))
    #sub_volt = pyplot.subplot(gs[16:20,0:12])
    sub_volt = pyplot.subplot(gs[6:11,1:12])
    for i,v in enumerate(volt_e):
        if v<-59 and volt_e[i-1]>-55:
            #pyplot.vlines((i-1)*tres,-50,-40)
            volt_e[i-1] = -40
            #volt_e[i] = -40
            #volt_e[i+1] = -40
            #print i
    pyplot.plot(net.mon_volt_e.times/ms, volt_e,'black')

    volt_sp_x = nrn_spikes*1000 
    volt_sp_y = -40*numpy.ones(len(volt_sp_x)) 
    
    pyplot.plot(volt_sp_x,volt_sp_y,'o',color='red')
    
    #for i,v in enumerate(volt_e):
        #if volt_e[i-1]>-45:
            #pyplot.plot((i-1)*tres,-40,'o',color='red')

    ylim=[-62,-39]
    yticks=[-60,-50,-40]
    #remove_spines(sub_volt,xlim=xlim,ylim=ylim,
                #xticks=xticks,yticks=yticks,
                #ylabel='V [mV]',legend= [],keepx=True)
    remove_spines(sub_volt,xlim=xlim,ylim=ylim,
                xticks=[],yticks=yticks,
                ylabel='V [mV]',legend= [])
    #sub_volt.set_xticklabels(xticks_lab, size=18)
    #pyplot.xlabel('Time [ms]',size=18)
    sub_volt.set_ylabel('',size=20)
    sub_volt.text(ylab_xposit,-40,'Voltage [mV]',rotation = 'vertical',size=20)
    sub_volt.set_yticks(yticks)
    sub_volt.set_yticklabels(yticks,size=20)
    sub_volt.yaxis.set_tick_params(width=1,length=5)

    sub_volt.plot(numpy.arange(xlim[0],xlim[1]),
                                -50.*numpy.ones(1000),'-.',color = 'gray')
    
    sub_volt.text(xticks[0]-130,-40.,'B',size=28,color='black')
    #sub_volt.text(xticks[-1]-00,-55.,'$V_{m}$',color='black',size=18)

    # draw vertical lines between spikes in raster and voltage trace

    linii_nekvi=[]
    curr_sp_y = 400.
    for i in range(len(volt_sp_x)):
        #linii_nekvi.append(get_straightline(figure,sub_rast,sub_volt,
        #        raster_sp_x[i],raster_sp_y[i],volt_sp_x[i],volt_sp_y[i]))
        linii_nekvi.append(get_straightline(figure,sub_rast,sub_curr,
                raster_sp_x[i],raster_sp_y[i],volt_sp_x[i],curr_sp_y))
    '''
    transFigure = figure.transFigure.inverted()
    c1x = raster_sp_x[0]
    c1y = raster_sp_y[0]
    c2x = volt_sp_x[0]
    c2y = volt_sp_y[0]
    coord1 = transFigure.transform(sub_rast.transData.transform([c1x,c1y]))
    coord2 = transFigure.transform(sub_volt.transData.transform([c2x,c2y]))
    line = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                        transform=figure.transFigure,color='gray',alpha=.9)

    print c1x,c1y,c2x,c2y
    print coord1, coord2
    '''
    figure.lines = linii_nekvi

    loc='tfigs/rcv6/'
    prefix = 'rccv_'
    fname = 'prf' + str(net.pr_ee) + str(net.pf_ee) + '_' + \
            str(-nrn) + '_int' + str(xlim[0])
    figure.savefig(loc+prefix+fname+'.eps', format='eps')
    figure.savefig(loc+prefix+fname+'.png', format='png')
    figure.savefig(loc+prefix+fname+'.pdf', format='pdf')

    #pyplot.close()
    pyplot.show()

def plot_gj_currents(net, nrn = 0):
    pyplot.figure()
    for n in range(len(net.nrn_meas_i)):
        pyplot.subplot(len(net.nrn_meas_i),1,n+1)
        pyplot.plot(net.mon_gjcurr.times/ms, 
                            net.mon_gjcurr[net.nrn_meas_i[n]]/pA)
        pyplot.ylabel('Igap [pA]')
        pyplot.xlabel('t [ms]')
        pyplot.legend(['IN %d'%net.nrn_meas_i[n]])

def plot_weights(net):
    """plot histograms of the weights before each balancing"""
    pyplot.figure()
    h_max = numpy.max(numpy.array([max(w) for w in net.weights]))
    h_min = numpy.min(numpy.array([min(w) for w in net.weights]))
    print 'aa',h_max
    #h_max = net.g_max/siemens
    bins = numpy.linspace(0.,h_max,100)
    for i in range(len(net.weights)):
        pyplot.subplot(len(net.weights),1,i+1)
        pyplot.hist(net.weights[i],bins)


def plot_gr_fr(net,ch=0,grs='all', wbin=10*ms):
    ''' plots the FR of all the assemblies in chain ch '''
    figure = pyplot.figure(figsize=(16.,12.))
    if grs == 'all':
        for gr in range(net.n_ass):
            pyplot.plot(net.mon_rate_gr[ch][gr].smooth_rate(wbin))


def plot_gr_fr2(nn, ps=0, wbin=.2, ngroups=None):
    ''' 
        plots the FR of all the assemblies in chain ch 
        wbin is bin size in ms

    '''

    frs = calc_spikes.make_fr_from_spikes(nn, ps=ps, w=wbin)
    time = numpy.arange(len(frs[0]))*wbin

    figure = pyplot.figure(figsize=(16., 12.))

    if ngroups:
        groups_to_show = ngroups
    else:
        groups_to_show = len(frs)
     
    for gr in range(groups_to_show):
        pyplot.plot(time, frs[gr])

    pyplot.xlabel('Time [ms]')
    pyplot.ylabel('Firing rate [1/s]')


def plot_mean_curr_act(nn, tl, ps=0, gr=0, dur_stim=100, dur_pre=10, wbin=.2,
                        sigma=.5):
    '''
       tl is list with time points that a stimulation has been initiated 
       wbin time resolution in ms

    '''
    import numpy as np
    frs = calc_spikes.make_fr_from_spikes(nn, ps=ps, w=wbin)
    mean_fr = np.zeros((dur_stim+dur_pre)/wbin+1)
    for t in tl:
        tindex = t/wbin
        mean_fr += frs[0][tindex-dur_pre/wbin-1: tindex+dur_stim/wbin]

    mean_fr /= len(tl)
    
    time = np.arange(-dur_pre, dur_stim+wbin, wbin)
    pyplot.plot(time, mean_fr)
    pyplot.plot(time, calc_spikes.gaus_smooth(mean_fr, w=wbin, sigma=sigma))
    return mean_fr 
    

def plot_inh_power(net, ch=0, t=0*second, shift=256, nfft=2048):
    [fp,pp]= periodogram(net.mon_rate_i.rate[-t/ms/(net.m_ts/ms):-1], 
                                                    shift, nfft, net.m_ts)
    pyplot.figure()
    #pyplot.subplot(221)
    pyplot.loglog(fp,pp)
    pyplot.ylabel('Power spectrum')
    pyplot.title('inh population')

def plot_power(net, ch=0, t=5*second, shift=256, nfft=2048):
    """plot power spectrum of some neurons"""
    # compare power before and after balancing!
    [fp,pp]= periodogram(net.rate_Me.rate[-t/ms/(net.m_ts/ms):-1], 
                                                    shift, nfft, net.m_ts)
    [fg1,pg1]= periodogram(net.rate_Mg[ch][0].rate[-t/ms/
                                (net.m_ts/ms):-1], shift, nfft, net.m_ts)
    [fg2,pg2]= periodogram(net.rate_Mg[ch][net.num_ass/2].rate[-t/ms/
                                (net.m_ts/ms):-1], shift, nfft, net.m_ts)
    [fg3,pg3]= periodogram(net.rate_Mg[ch][net.num_ass-1].rate[-t/ms/
                                (net.m_ts/ms):-1], shift, nfft, net.m_ts)
    pyplot.figure()
    pyplot.subplot(221)
    pyplot.loglog(fp,pp)
    pyplot.ylabel('Power spectrum')
    pyplot.title('exc population')
    pyplot.subplot(222)
    pyplot.loglog(fg1,pg1)     
    pyplot.title('first assembly')
    pyplot.subplot(223)
    pyplot.loglog(fg2,pg2)
    pyplot.ylabel('Power spectrum')
    pyplot.title('mid assembly')
    pyplot.xlabel('Freq [Hz]')
    #pyplot.xlim([0, 2000])
    pyplot.subplot(224)
    pyplot.loglog(fg3,pg3)     
    pyplot.xlabel('Freq [Hz]')
    #pyplot.xlim([0, 2000])
    pyplot.title('last assembly')

def plot_avalanches(net, ch=0, n_bins=1000, delta_0=1, cut_off=30000):

    [dur_l, size_l] = get_avalanches(net.rate_Me.rate,delta_0,cut_off)
    
    pyplot.figure()
    pyplot.subplot(211)
    a = numpy.histogram(size_l,numpy.arange(n_bins))
    pyplot.loglog(a[0])
    pyplot.subplot(212)
    pyplot.loglog(numpy.histogram(dur_l)[0])
    
    groups = [0,net.num_ass/2,net.num_ass-1]
    pyplot.figure()
    for i, gr in enumerate(groups):
        # some weirdness with the rate, have to normalize it!
        rate = net.rate_Mg[ch][gr].rate/min(net.rate_Mg[ch][gr].rate[
                    net.rate_Mg[ch][gr].rate>0]) 
        [dur_l, size_l] = get_avalanches(rate, delta_0, cut_off)

        pyplot.subplot(2,len(groups),i+1)
        a = numpy.histogram(size_l,numpy.arange(n_bins))
        pyplot.loglog(a[0])
        pyplot.subplot(2,len(groups),len(groups)+i+1)
        pyplot.loglog(numpy.histogram(dur_l)[0])
    
    #pyplot.subplot(212)
    #pyplot.loglog(numpy.histogram(delta_l)[0])

def show():
    pyplot.show()

def plot_fr_cv_syn_distr(nn):
    import numpy as np
    t0,t1 = 20,25
    frs = calc_spikes.get_group_fr_distr(nn,0,9,t0,t1)
    cvs = calc_spikes.get_group_cv_distr(nn,0,9,t0,t1)
    syncs = calc_spikes.get_group_synch_distr(nn,0,9,t0,t1)

    frs = frs[np.isfinite(frs)]
    cvs = cvs[np.isfinite(cvs)]
    synchs = syncs[np.isfinite(syncs)]


    figure = pyplot.figure(figsize=(6.,9.))

    subp1 = pyplot.subplot(311)
    pyplot.hist(frs,np.arange(0,20.1,.5),weights=np.zeros(frs.size)+1./frs.size)
    pyplot.ylabel('Relative frequency',size=14)
    pyplot.xlabel('Firing rate [spikes/sec]',size=14)
    yticks = [0,.08,.16]
    pyplot.yticks(yticks)
    remove_spines_only(subp1,True)

    subp2 = pyplot.subplot(312)
    pyplot.hist(cvs,np.arange(0,4.01,.1),weights=np.zeros(cvs.size)+1./cvs.size)
    pyplot.ylabel('Relative frequency',size=14)
    pyplot.xlabel('Coefficient of variation',size=14)
    yticks = [0,.05,.1]
    pyplot.yticks(yticks)
    remove_spines_only(subp2,True)
    
    subp3 = pyplot.subplot(313)
    pyplot.hist(synchs,np.arange(-0.2,.301,.01),
                weights=np.zeros(synchs.size)+1./synchs.size)
    pyplot.ylabel('Relative frequency',size=14)
    pyplot.xlabel('Synchrony',size=14)
    yticks = [0,.3,.6,.9]
    pyplot.yticks(yticks)
    remove_spines_only(subp3,True)

    figure.tight_layout()

