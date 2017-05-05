def get_avalanches(rate, delta_0=1, cut_off=30000):
    '''
    simple function which extracts avalanches from the FR monitor
    delta_0 is the length of the silent interval
    returns lists with the duration and size of the avalanches
    '''
    s=0
    size_l=[]   # list with avalanche sizes
    dur_l=[]    # with their duration
    delta_l=[]  # length of silent periods

    delta=0
    dur=0
    #for i in self.rate_Me.rate[cut_off:]:
    for i in rate[cut_off:]:
        if i:
            s+=i
            dur+=1
        elif s:
            delta+=1
            if delta>=delta_0:
                size_l.append(s)
                dur_l.append(dur)
                s=0
                dur=0
                delta=0    

    return dur_l, size_l

