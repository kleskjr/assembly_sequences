import numpy
from brian import second,ms
from matplotlib import pyplot
from scipy.stats import kurtosis
from peakdetect import peakdet

def rec_aver(st,w=.5,dt=.1):
    return numpy.convolve(st, numpy.ones(w/dt)/w*1000.,'same')
    #return numpy.convolve(st, numpy.ones(w/dt)/w*dt,'same')

def get_gr_st(nn,ch=0,g1=0):
    ''' get group spike train
        averages over the size of the group'''
    dt = nn.m_ts/ms
    strain = numpy.zeros(nn.network.clock.end/second*1000/dt)
    for j in range(nn.s_ass):
        stimes = nn.mon_spike_e[nn.P_ass_index[ch][g1][j]]*1000./dt
        strain[stimes.astype('Int32')]+=1
    return strain/nn.s_ass

def get_spike_train(times,leng=10,t_res=.001):
    ''' get spike train, all unit should be in secs '''
    spike_tr = numpy.zeros((leng*1000+1)/t_res)
    spike_tr[(times/t_res).astype('Int32')]=1
    return spike_tr

def get_pairwise_corr(nn,gr_spikeM,w=1.):
    mean_corr=0.
    for m1 in range(nn.s_ass):
        print m1
        s1 = get_spike_train(gr_spikeM[m1])
        for m2 in range(m1):
            s2 = get_spike_train(gr_spikeM[m2])
            mean_corr += numpy.dot(s1,s2)
    return mean_corr/(nn.s_ass*(nn.s_ass-1.)/2.) 



def corr(nn,g1=0,g2=1,halfp = 500.,tstim=18.):
    ''' just correlates the group activity of 2 groups around tstim[sec]
    halfp [ms]...'''
    ch=0
    dt = nn.m_ts/ms
    t0 = tstim*1000-halfp
    t1 = tstim*1000+halfp

    s1 = get_gr_st(nn,ch,g1)
    s2 = get_gr_st(nn,ch,g2)

    p1 = s1[t0/dt:t1/dt]
    p2 = s2[t0/dt:t1/dt]
   
    a1 = rec_aver(p1,5.) - rec_aver(p1,5.).mean()
    a2 = rec_aver(p2,5.) - rec_aver(p2,5.).mean()
    
    c2 = numpy.correlate(a1,a2,'same')/(a1.std()*a2.std())/(2*halfp/dt)
    return c2

def get_ccg(nn, tstim =19.):
    '''
    calculates cross-correliograms between assemblies
    plot them in a low-diagonal matrix
    '''
    ch=0
    halfp = 500.
    dt = nn.m_ts/ms
     
    t = numpy.arange(0,2*halfp,dt)-halfp
    pyplot.figure()
    for g1 in range(nn.n_ass):
        for g2 in range(g1+1):
            c = corr(nn,g1,g2,halfp,tstim)
            pyplot.subplot(nn.n_ass, nn.n_ass, g1*nn.n_ass+g2+1)
            pyplot.axvline(0,-.1,1.,c='red')
            pyplot.plot(t,c)
            pyplot.ylim([-.1,1.])
            pyplot.xlim([-150,150.]) # limit the plot in +-100ms
            pyplot.axis('off')
    #pyplot.show()

def get_power(nn, ch=0, t=50*second, shift=256, nfft=2048):
    from periodogram import periodogram
    [fp,pp]= periodogram(nn.rate_Me.rate[-t/ms/(nn.m_ts/ms):-1], 
                                                    shift, nfft, nn.m_ts)
    [fg3,pg3]= periodogram(nn.mon_rate_Mg[ch][nn.n_ass-1].rate[-t/ms/
                                (nn.m_ts/ms):-1], shift, nfft, nn.m_ts)
    return pp,pg3

def get_activ(nn,g1=0,g2=1,halfp=200,tstim=19.):
    ch=0
    dt = nn.m_ts/ms

    t0 = tstim*1000-halfp
    t1 = tstim*1000+halfp

    max_activ = []
    disper = []
    for g in range(nn.n_ass):
        s = get_gr_st(nn,ch,g)
        p = rec_aver(s[t0/dt:t1/dt],5.)
        p_th = p
        thresh = 34.
        #p_th[p_th<thresh] = 0.
        peak_y = p_th.max()
        max_activ.append(peak_y)
        peak_x = t0 + p_th.argmax()*dt 
        stimes = [] 
        t0_dis = peak_x-10
        t1_dis = peak_x+10
        for j in range(nn.s_ass):
            times = nn.mon_spike_e[nn.P_ass_index[ch][g][j]]*1000.
            stimes.extend(filter(lambda x:x>t0_dis and x<t1_dis ,times))
        print peak_x, peak_y

        disper.append((numpy.array(stimes) - peak_x).std())
        #disper.append(get_pulse(stimes,peak_x,1.0))

        pyplot.subplot(313)
        pyplot.plot(p)
    
    pyplot.subplot(311)
    pyplot.plot(max_activ)
    pyplot.subplot(312)
    pyplot.plot(disper)
    #print disper
    #print peak_x
    #return max_activ, disper
    return stimes

def get_actpeaks(nn, thresh=30):
    ch=0
    act_peaks=[]
    for gr in range(nn.n_ass):
        r = nn.mon_rate_Mg[ch][gr].smooth_rate(width=2.0*ms)
        act_peaks.append(peakdet(r,thresh)[0])
    return act_peaks

def get_singleprop(peaks,t_sec=50,evoked_p=[]):
    ''' get the depth of propagation at time t_sec '''
    t_res = .1
    t = t_sec*1000./t_res
    dt = 15./t_res
    cnt = 0

    for i,c in enumerate(peaks):
        for j,v in enumerate(c):
            fl=False
            #print v,t
            if (v[0]-t)<dt and v[0]>t:
                evoked_p.append(v)
                t = v[0]
                cnt+=1
                fl=True
                break
        if fl==False: break
    return cnt

def get_meanprop(peaks, first=50, last=100, stim_int = 1):
    ''' get average propagation depth from all the stimulations in run state'''
    mean_l = 0.
    evoked_p = []
    for t in xrange(first, last):
        c = get_singleprop(peaks,t,evoked_p)
        mean_l +=c 
    mean_l/= float((last-first)/stim_int) # 50 stimulation in [50, 100] sec 
    return evoked_p, mean_l

def get_spontprop(peaks, evoked, first=50, last = 100):
    '''checks the number of propagating waves, not counting the evoked ones 
        starts checking from the last group with depth depth'''
    t_res = .1
    depth = 5
    n_spont= 0

    for v in peaks[-1]:
        if v[0] < first*1000/t_res: continue
        if v in evoked: continue
        if v[0] > last*1000/t_res: break
        #print v
        t=v[0]
        for i in range(depth):
            flg, t = get_prev(peaks, evoked,i+2,t) 
            if flg == False: break
        if flg:     n_spont+=1
    return n_spont
    
def get_propagation(peaks, first=50, last = 100):
    '''uaa '''
    evoked, mean_evoked = get_meanprop(peaks, first, last)
    n_spont = get_spontprop(peaks,evoked,first, last)    
    return mean_evoked, n_spont

def get_prev(peaks, evoked, i,t):
    '''returns flag showing whether there is a peak in the preceding group'''
    t_res = .1
    dt = 15./t_res
    flg = False
    for v in peaks[-i]:
        if (t-v[0])>dt or v[0]>t: continue
        if v in evoked: continue
        flg = True
        #print 'vvvvv', i, v
        if flg ==True: break
    return flg, v[0]
            
def get_group_cv(gr_mon,start,stop):
    ''' get cv from a few representitive neurons''' 
    cv=[]
    for i in range(gr_mon.source.N):
        sm = gr_mon[i]
        times = sm[sm>=start]
        times = times[times<=stop]
        isi = numpy.diff(times)
        cv.append(isi.std()/isi.mean())
    return cv

def get_cv(nn,start=10.,stop=100):
    mean_gr_cv = []
    for gr_mon in nn.spike_Mgr:
        cv_l = get_group_cv(gr_mon,start,stop)
        # get rid of nans and estimate mean CV
        m = numpy.isfinite(cv_l)
        mean_gr_cv.append(numpy.array(cv_l)[m].sum()/m.sum())

    return mean_gr_cv

def get_fano(nn,start=10.,stop=100.):
    stop = min(stop, nn.network.clock.end/second)
    if start > stop : return -1
    bins = numpy.arange(start,stop)
    mean_fano=[] 

    for gr_mon in nn.spike_Mgr:
        ff=[]
        for i in range(gr_mon.source.N):
            hst = numpy.histogram(gr_mon[i],bins)
            ff.append(hst[0].var()/hst[0].mean())
        
        # mean without counting the nans
        m = numpy.isfinite(ff)
        mean_fano.append(numpy.array(ff)[m].sum()/m.sum())

    return mean_fano

def get_pulse(stimes,peak_x, t_theta = 2.0):
    ''' gets the mean and standart deviation of a pulse packet(Gewaltig01)
        t_theta: max distance [ms] which can separate spikes from the packet'''
    x = numpy.array(stimes)-peak_x 
    
    p = x[x>0] # play just with the positive (right to the peak)
    t0 =0
    for t in p:
        if (t-t0)>t_theta:
            p = p[p<t]
            break
        t0=t
    n = x[x<0] # and negative
    t0 =0
    for t in n[::-1]:
        if (t0-t)>t_theta:
            n = n[n>t]
            break
        t0=t
    l=[]
    l.extend(p)
    l.extend(n)
    
    #y=numpy.array(l)
    y=numpy.array(x)
    s = (y-y.mean()).std()
    kurt = ((y-y.mean())**4).mean()/s**4-3.
    kurt2 = kurtosis(y)
    #print len(stimes), len(y), s
    print kurt, 
    return s
    #return kurt

def plot_prop():
    import os
    import pickle
    from matplotlib import pyplot

    pr_l = numpy.arange(.03,.145,.01)
    pf_l = numpy.arange(.02,.165,.01)
    
    imag = numpy.zeros((len(pr_l), len(pf_l)))
    for i,pr in enumerate(pr_l):
        for j,pf in enumerate(pf_l):

            fname = 'datf/pr' + str(pr) + 'pf' + str(pf) + '.pck'
            print fname
            if os.path.exists(fname)==False:
                fname = 'datf/pr' + str(pr)[1:] + 'pf' + str(pf) + '.pck'
                if os.path.exists(fname)==False:
                    fname = 'datf/pr' + str(pr)[1:] + '0pf' + str(pf) + '.pck'
                    if os.path.exists(fname)==False:
                        print 'no f.'
                        continue 
            f = open(fname)
            l = pickle.load(f)
            f.close()
            imag[i,j] = get_meanprop(l)[1]

    pyplot.imshow(imag,origin='lower', extent=[.03,.15,.01,.15])
    pyplot.xlabel('p_r')
    pyplot.ylabel('p_f')
    return imag

def plot_smth():
    import os
    import pickle
    from matplotlib import pyplot

    pr_l = numpy.arange(.00,.131,.01)
    pf_l = numpy.arange(.01,.131,.01)
    #gc_l = numpy.arange(.0005,.01,.0005)
    gc_l = numpy.arange(.002,.032,.002)
    #gc_l = numpy.arange(.004,.0305,.001)
    
    a_x = pr_l
    #a_x = gc_l
    a_y = pf_l
    
    imag_evoked_prop= numpy.zeros((len(a_y), len(a_x)))
    imag_spont_prop= numpy.zeros((len(a_y), len(a_x)))
    imag_cv = numpy.zeros((len(a_y), len(a_x)))
    imag_fano = numpy.zeros((len(a_y), len(a_x)))
    #for j,gc in enumerate(gc_l):
    p_rand = 0.01
    gc = 0.0125

    for i,pf in enumerate(pf_l):
        for j,pr in enumerate(pr_l):
            #pf=pr
            #pr=0.0
            #fname= 'datf_sfc1/r1g'+str(gc)+ 'pr'+str(pr) + 'pf'+str(pf) + '.pck'
            #fname= 'datf_sfc2/r1g'+str(gc)+ 'pr'+str(pr) + 'pf'+str(pf) + '.pck'
            #fname = 'datf2/g'+str(gc)+ 'pr'+str(pr) + 'pf'+str(pf) + '.pck'
            fname = 'data/datf_pr_vs_pf/g'+str(gc)+'r'+str(p_rand)+\
                                'pr'+str(pr) + 'pf'+str(pf) + '.pck'
            print fname
            if os.path.exists(fname)==False:
                print 'no f.'
                continue 
            f = open(fname)
            l = pickle.load(f)
            f.close()
            '''
            n_evoked, n_spont= get_spontprop(l[0])
            imag_evoked_prop[i,j] = n_evoked
            imag_spont_prop[i,j] = n_spont
            '''
            imag_evoked_prop[i,j] =l[0][0]
            imag_spont_prop[i,j] = l[0][1]
            imag_cv[i,j] = 1/3.*(l[1][2] + l[1][2]+l[1][2])
            imag_fano[i,j] = 1/3.*(l[2][2] + l[2][2]+l[2][2])

    ratio_plot = (a_x[-1]-a_x[0])/ (a_y[-1]-a_y[0])

    figure = pyplot.figure()
    pyplot.imshow(imag_evoked_prop,origin='lower',aspect=ratio_plot,
                                extent=[a_x[0],a_x[-1],a_y[0],a_y[-1]])
    pyplot.title('Mean evoked propagation')
    #pyplot.xlabel('Conductance')
    pyplot.xlabel('pr')
    pyplot.ylabel('pf')
    pyplot.colorbar()
    figure.savefig('some_plots3/120815_pr_vs_pf_evoked.eps', format = 'eps')
    figure = pyplot.figure()
    pyplot.imshow(imag_spont_prop,origin='lower',aspect=ratio_plot,
                                extent=[a_x[0],a_x[-1],a_y[0],a_y[-1]])
    pyplot.title('Number of spontan waves')
    pyplot.xlabel('Conductance')
    pyplot.ylabel('p(conn)')
    pyplot.xlabel('pr')
    pyplot.ylabel('pf')
    pyplot.colorbar()
    figure.savefig('some_plots3/120815_pr_vs_pf_spont.eps', format = 'eps')
    figure = pyplot.figure()
    pyplot.imshow(imag_cv,origin='lower',aspect=ratio_plot,
                                extent=[a_x[0],a_x[-1],a_y[0],a_y[-1]])
    pyplot.title('CV')
    pyplot.colorbar()
    pyplot.xlabel('Conductance')
    pyplot.ylabel('p(conn)')
    pyplot.xlabel('pr')
    pyplot.ylabel('pf')
    figure.savefig('some_plots3/120815_pr_vs_pf_cv.eps', format = 'eps')
    figure = pyplot.figure()
    pyplot.imshow(imag_fano,origin='lower',aspect=ratio_plot,
                                extent=[a_x[0],a_x[-1],a_y[0],a_y[-1]])
    pyplot.title('FanoF')
    pyplot.colorbar()
    pyplot.xlabel('Conductance')
    pyplot.ylabel('p(conn)')
    pyplot.xlabel('pr')
    pyplot.ylabel('pf')
    figure.savefig('some_plots3/120815_pr_vs_pf_ff.eps', format = 'eps')
    
    return imag_evoked_prop,imag_spont_prop,imag_cv,imag_fano
    #return imag_evoked_prop,imag_spont_prop

def calc_corr(nn, g1=0, g2=1):
    '''eventually correlates 2 groups'''
    ch = 0
    dt = nn.m_ts/ms
    
    f1 = nn.mon_rate_Mg[ch][g1].smooth_rate(width=.5*ms)
    f2 = nn.mon_rate_Mg[ch][g2].smooth_rate(width=.5*ms)

    s1 = get_gr_st(nn,ch,g1)
    s2 = get_gr_st(nn,ch,g2)

    #c1 = numpy.correlate(f1, f2, 'same')
    #c2 = numpy.correlate(rec_aver(s1,5.),rec_aver(s2,5.),'same')\
    #                                    /nn.network.clock.end
    t = numpy.arange(0,nn.network.clock.end/second*1000,dt)
    
    pyplot.figure()
    pyplot.subplot(211)
    #pyplot.plot(t,s1)
    pyplot.plot(t,rec_aver(s1,5.))
    pyplot.plot(t,rec_aver(s2,5.))

    pyplot.subplot(212)
    #pyplot.plot(t,c1)
    #pyplot.plot(t-t[-1]/2,c2)
    pyplot.show()


def calc_corri_old_slow(nn):
    ''' calculates the correlations into a group
    .
    ..
    ...'''

    tau = 5.
    dt = 10.


    for runwin in numpy.arange(0,nn.network.clock.end/second*1000,dt):
        for ch in range(1):#nn.num_ffchains):
            for i in range(1):#nn.n_ass): 
                ass_syn=0.

                for j in range(nn.s_ass):
                    #print nn.P_ass_index[ch][i][j]
                    stimes = nn.mon_spike_e[nn.P_ass_index[ch][i][j]]*1000.
                    #if j==0: print stimes
                    ass_syn += sum(abs(stimes-runwin)<tau) 
                ass_syn/=nn.s_ass
        print runwin, ass_syn
    
