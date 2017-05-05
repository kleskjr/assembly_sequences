import numpy
from peakdetect import peakdet

def get_peaks(rates,cmpf,thresh):
    '''returns peaks above some thresh given the rates of multiple groups'''
    peaks=[]
    for gr in range(len(rates)):
        r = numpy.interp(numpy.arange(len(rates[gr])*cmpf)/cmpf,\
                                numpy.arange(len(rates[gr])),rates[gr])
        peaks.append(peakdet(r,thresh)[0])
    return peaks

def get_singleprop(peaks,max_delay,min_delay,t_sec=50,evoked_p=[],
                                                detect_bursts=True):
    ''' return the depth of propagation at time t_sec '''
    t_res = 1.
    t = t_sec*1000./t_res-t_res
    cnt = 0

    fl=False
    for i,c in enumerate(peaks):
        # c is list of peaks in current group i
        for j,v in enumerate(c):
            fl=False
            if (v[0]-t)<max_delay and (v[0]-t)>min_delay:
                # if the peak is appropriate time window
                print 'uaaaaaaa',v[0]
                evoked_p.append(v)
                t = v[0]
                cnt+=1
                fl=True
                # check for nearby peaks
                if detect_bursts:
                    for j2,v2 in enumerate(c):
                        if (v2[0]>v[0] and abs(v2[0]-v[0])<20):
                            print 'sorry, you burst'
                            return cnt-1
                break # go to next group

        if fl==False: break # finish with this stimulation
    return cnt

def get_meanprop(peaks,first=50,last=100,max_delay=12,min_delay=1,
                                stim_int=1,detect_bursts=True):
    '''
        estimates how far the wave propagates on average
        make it to return [mean std]
    '''
    mean_l = 0.
    evoked_p = []
    for t in xrange(first, last):
        c = get_singleprop(peaks,max_delay,min_delay,t,evoked_p,detect_bursts)
        mean_l +=c 
    mean_l/= float((last-first)/stim_int) # 50 stimulation in [50, 100] sec 
    return mean_l

def get_spont(peaks,first=40,last=60,max_delay=12,min_delay=1,depth=5,
                                                    detect_bursts=True):
    '''
        gets the mean number of spontaneous waves per second
        it starts looking at the peaks in the last group
        and dives 'depth' groups before that to track the wave
    '''
    n_spont = 0.
    t_lastdet = 0.
    for v in peaks[-1]: #start from last -1 (if control -2)
        if v[0] < first*1000: continue
        if v[0] > last*1000: break
        if detect_bursts and ((v[0]-t_lastdet)<20):
            t_lastdet = v[0]
            continue
        t = v[0]
        t_lastdet = v[0]
        flg = False
        for i in range(depth):
            # -1 last, tracking back starts at -2 (if control is in, then -3)
            flg,t = get_prev(peaks[-i-2],t,max_delay,min_delay) 
            if flg==False: break
        if flg: n_spont+=1
    return float(n_spont)/(last-first)
    #return n_spont

def get_prev(gr_peaks, t, max_delay,min_delay):
    '''returns flag showing whether there is a peak in the preceding group'''
    flg = False
    v = [0,0]
    v_t = v[0]
    for v in gr_peaks:
        if (t-v[0])>max_delay or (t-v[0])<min_delay: continue
        flg = True
        v_t=v[0]
        if flg ==True: break
    return flg, v_t


