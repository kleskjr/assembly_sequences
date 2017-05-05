import numpy

def get_spike_train(times, dur=40, w=1., start_t=90):
    '''
        get spike train of times vector
        start_t and dur are in seconds 
        w is the resolution (width of bin) is in ms

    '''
    w = w / 1000.
    spike_train = numpy.histogram(times,
                        numpy.arange(start_t, start_t+dur+w, w))[0]
    return spike_train


def get_coher(spikeM, dur=40, w=1, fraq=.1, start_t=90):
    '''
        returns coherence within groups
    '''
    coher = 0.
    n_corr = 0 # number of non nan correlation (for averaging)
    num_st  = int(fraq*len(spikeM))
    for m1 in range(num_st):
        st1 = get_spike_train(spikeM[m1],dur,w,start_t)
        if sum(st1):
            for m2 in range(m1):
                st2 = get_spike_train(spikeM[m2],dur,w,start_t)
                #c =  numpy.dot(st1,st2)/(sum(st1)*sum(st2))
                c = numpy.corrcoef(st1,st2)[0][1]
                #if numpy.isnan(c): print c,len(st1),len(st2),sum(st1),sum(st2)
                if numpy.isnan(c)==False:
                    coher += c
                    n_corr+=1
    return coher/n_corr


def get_cv(spikeM,start=90,stop=130):
    mean_gr_cv = []
    for gr_mon in spikeM:
        cv_l = get_group_cv(gr_mon,start,stop)
        m = numpy.isfinite(cv_l)# get rid of nans and estimate mean CV
        mean_gr_cv.append(numpy.array(cv_l)[m].sum()/m.sum())
    return mean_gr_cv


def get_group_cv(gr_mon,start,stop):
    ''' get cv from a few representitive neurons''' 
    cv=[]
    for i,sm in enumerate(gr_mon):
        times = sm[sm>=start]
        times = times[times<=stop]
        isi = numpy.diff(times)
        cv.append(isi.std()/isi.mean())
    return cv


def get_fano(spikeM,start=90.,stop=130.,w=1.):
    if start > stop : return -1
    bins = numpy.arange(start,stop,w)
    mean_fano=[] 

    for gr_mon in spikeM:
        ff=[]
        for i in range(len(gr_mon)):
            hst = numpy.histogram(gr_mon[i],bins)
            ff.append(hst[0].var()/hst[0].mean())
        
        # mean without counting the nans
        m = numpy.isfinite(ff)
        mean_fano.append(numpy.array(ff)[m].sum()/m.sum())
    return mean_fano


def sptimes2spikes(sptimes):
    '''
    r=[]
    for i, sp in enumerate(sptimes):
        for t in sp:
            r.append((i,t))
    '''
    r = [(i,t) for i,sp in enumerate(sptimes) for t in sp]
    #r.sort(key=lambda tup: tup[1]) #seems they should not be necessary sorted
    return r


def get_spike_times_ps(nn, n_ps=0, frac=1., permutate=False, exc_nrns=True,
                    dummy_ass=False, dummy_ass_frac=None, pick_first=True):
    '''
        used for bb.raster_plot_times
        gets the spike times of the neurons participating in PS n_ps
        ordered according to the phase sequence arrangement
        frac is the fraction of neurons from each group to be returned
        permutate: whether to permutate the elements of each group, useful
        for plotting purposes, when just a fraction of the first neurons are
        stimulated, and we do not need to plot just them
        dummy_ass is True if random neurons (outsie of the PS) are to be used
        pick_first to pick the first frac neurons from a group; if False
            then every 1/frac neuron is picked (made for the contin ASS case)

    '''
    if exc_nrns:
        index = nn.p_ass_index
        mon_spike = nn.mon_spike_e
    else:
        index = nn.p_assinh_index
        mon_spike = nn.mon_spike_i

    sp=[]
    n=0
    for gr in index[n_ps]:
        if permutate:
            gr_neurons = numpy.random.permutation(gr)
        else:
            gr_neurons = gr
            if pick_first:
                for nrn in gr_neurons[0:frac*len(gr)]:
                    #print nrn
                    for t in mon_spike[nrn]:
                        sp.append((n,t))
                    n+=1
            else:
                for nrn in gr_neurons[::1/frac]:
                    #print nrn
                    for t in mon_spike[nrn]:
                        sp.append((n,t))
                    n+=1
                 
    # optimize if feel bored! 
    #r = [(i,t) for i,sp in enumerate(sptimes) for t in sp]

    if dummy_ass:
        if dummy_ass_frac==None:
            dummy_ass_frac=frac
        for nrn in nn.dummy_ass_index[n_ps][0:int(dummy_ass_frac*nn.s_ass)]:
            for t in mon_spike[nrn]:
                sp.append((n,t))
            n+=1

    return sp


def get_all_spikes(nn):
    '''
        get the spike times from spike monitor and put them in a
        list of tupples that is returned.

        This is to be saved and plotted later on.
    
    '''
    sp=[]
    for nrn in range(len(nn.mon_spike_e.source)):
        for t in nn.mon_spike_e[nrn]:
            sp.append((nrn, t))
    return sp


def get_alpha_sigma(nn,t,t_pre=.005,t_post=.025,ps=0,n_spikes=True):
    '''
        get alpha and sigma of a wave evoked at time t
        [t-t_pre,t+t_post] is the time interval in which we measure
        ps: which phase sequence
        n_spikes: whether to return alpha as total number of spikes in
            the group, else returns number of neuron spiking
    '''
    alpha = []
    sigma = []

    for gr in range(nn.n_ass):
        t0 = t-t_pre
        t1 = t+t_post
        times, n_spiking_nrn =get_tot_spike_times(nn,t0,t1,ps=ps,gr=gr)
        t = numpy.mean(times)           # mean spike time (the peak, kind of)
        sigma.append(numpy.std(times)*1000.)  # std of spike times in ms

        if n_spikes:
            alpha.append(len(times)/float(nn.s_ass))  # number of spikes in [t0,t1]
        else:
            alpha.append(n_spiking_nrn/float(nn.s_ass))

        #print t,alpha[-1],sigma[-1]
    return alpha, sigma


def get_alpha_sigma_dummy(nn,t,t_pre=.005,t_post=.025,ps=0,n_spikes=True):
    ''' hui '''
    t0 = t-t_pre
    t1 = t+t_post
    times, n_spiking_nrn =get_tot_spike_times_dummy(nn,t0,t1)
    t = numpy.mean(times)           # mean spike time (the peak, kind of)

    sigma = numpy.std(times)*1000.  # std of spike times in ms
    if n_spikes:
        alpha = len(times)/float(nn.s_ass)  # number of spikes in [t0,t1]
    else:
        alpha = n_spiking_nrn/float(nn.s_ass)

    return alpha, sigma


def get_tot_spike_times_dummy(nn,t0,t1,n_dummynrns=500):
    '''
      
    '''
    times = []
    n_spiking_nrn=0.
    for nrn in range(n_dummynrns):
        t = nn.mon_spike_d[nrn]
        t = t[t>t0]
        t = t[t<t1]
        times.extend(t)
        if t.any(): n_spiking_nrn +=1
    #print numpy.mean(times), numpy.std(times)
    return times, n_spiking_nrn


def get_tot_spike_times(nn,t0,t1,ps=0,gr=0):
    '''
        returns a list of all the spike times of neurons in group gr
        in sequence ps, in the time window t0,t1
        w: time resolution, default 1 ms
    '''
    times = []
    n_spiking_nrn=0.
    for nrn in nn.p_ass_index[ps][gr]:
        t = nn.mon_spike_e[nrn]
        t = t[t>t0]
        t = t[t<t1]
        times.extend(t)
        if t.any(): n_spiking_nrn +=1
    #print numpy.mean(times), numpy.std(times)
    return times, n_spiking_nrn


def get_tot_spike_train(nn,t0=12.,t1=13.,ps=0,gr=0,w=1.):
    '''
        get average spike train of phase sequence ps
        of group gr in the interval [t0,t1]
        w: time resolution, default 1 ms
    '''
    w=w/1000.   # ms to seconds
    tot_spike_train=numpy.zeros((t1-t0+w)/w)
    for nrn in nn.p_ass_index[ps][gr]:
        tot_spike_train += numpy.histogram(nn.mon_spike_e[nrn],
                        numpy.arange(t0,t1+w,w))[0]

    t= numpy.histogram(nn.mon_spike_e[nrn],numpy.arange(t0,t1+w,w))[1]
    return tot_spike_train,t


def make_fr_from_spikes(nn, ps=0, w=1, exc_nrns=True):
    from brian import second, ms
    '''
        Estimate FR for neural assemblies based on the
        spike monitor
        Used in case that assemblies are consisting of random neurons
        and can't be in a NeuroGroup and therefore, no rate monitor 
        can be attached

        w is probably the resolution in ms

    '''
    dur = nn.network.clock.t/second
    group_fr = []
    if exc_nrns:
        index = nn.p_ass_index
        mon_spike = nn.mon_spike_e
    else:
        index = nn.p_assinh_index
        mon_spike = nn.mon_spike_i

    for gr in range(nn.n_ass):
        fr = numpy.zeros(dur*1000/w)
        gr_size = len(index[ps][gr])
        for j in index[ps][gr]:
            fr += get_spike_train(mon_spike[j], dur=dur, start_t=0, w=w)
        # here multiplied to normalize
        group_fr.append(fr*1000./w/gr_size)
    return group_fr


def gaus_smooth(r, w=1, sigma=2):
    sfreq = 1./w
    gaussian = numpy.exp(-numpy.arange(-3*sigma+0*w, 3*sigma+1*w, w)**2 \
                        / (2*sigma**2))
    return numpy.convolve(r, gaussian, 'same')/sum(gaussian)


def get_group_fr_distr(nn,ps=0,gr=0,t0=0,t1=None,exc_nrns=True):
    '''
        estimates the fr of every neuron in ps, gr in the interval [t0,t1]
        exc_nrns: excitatory or inhibitory neurons?
    '''
    from brian import second, ms
    if not t0:
        t0=0
    if not t1:
        t1 = nn.network.clock.t/second
    frates = []
    for j in nn.p_ass_index[ps][gr]:
        frates.append(sum(nn.mon_spike_e[j][nn.mon_spike_e[j]>t0]<t1)/\
                        float(t1-t0))
    return numpy.array(frates)

    
def get_group_cv_distr(nn,ps=0,gr=0,t0=0,t1=None):
    '''
        estimates the cv of every neuron in ps, gr in the interval [t0,t1]
    '''
    from brian import second, ms
    if not t0:
        t0=0
    if not t1:
        t1 = nn.network.clock.t/second
    cvs = []
    for j in nn.p_ass_index[ps][gr]:
        isi = numpy.diff(nn.mon_spike_e[j][nn.mon_spike_e[j]
                        [nn.mon_spike_e[j]>t0]<t1])
        cvs.append(isi.std()/isi.mean())
    return numpy.array(cvs)

        
def get_group_synch_distr(nn,ps=0,gr=0,t0=0,t1=None,fraq=.1):
    '''
        estimates the cv of every neuron in ps, gr in the interval [t0,t1]
    '''
    from brian import second, ms
    if not t0:
        t0=0
    if not t1:
        t1 = nn.network.clock.t/second
    dur = t1-t0 #duration of spike trains
    w=1.        # 1 ms bin
    synchs = []
    for i,n1 in enumerate(nn.p_ass_index[ps][gr][:int(nn.s_ass*fraq)]):
        st1 = get_spike_train(nn.mon_spike_e[n1],dur,w,t0)
        for j in range(i):
            n2 = nn.p_ass_index[ps][gr][j]
            st2 = get_spike_train(nn.mon_spike_e[n2],dur,w,t0)
            synchs.append(numpy.corrcoef(st1,st2)[0][1])
    return numpy.array(synchs)

