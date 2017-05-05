#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import brian as bb
from brian import ms, second, Hz, mV, pA, nS, pF
#from np.random import rand,binomial
from time import time, asctime
import warnings
import nekvo
import sys

### some custom modules
import plotter
import calc_spikes

### some brian optimizations
# import brian_no_units
#bb.globalprefs.set_global_preferences(useweave=True)
#bb.globalprefs.set_global_preferences(usecodegen=True,
#                   usenewpropagate=True, usestdp=True)
g_l = 10.*nS
C_m = 200*pF
v_r = -60.*mV
v_e = 0.*mV
v_i = -80.*mV
tau_m_exc = 20.*ms
tau_m_inh = 20.*ms
tau_inh = 10*ms
tau_fast_inh = 10*ms
tau_exc = 5.*ms
tau_stdp = 20.*ms
alpha = .2
g_min = 0*nS
g_max = 50*nS

eqs_exc = '''dv/dt = (g_l*(v_r-v)+Ie+Ii+I)/(C_m) : volt
            dge/dt = -ge/(tau_exc) : siemens
            dgi/dt = -gi/(tau_inh) : siemens
            Ie = ge*(v_e-v) : amp
            Ii = gi*(v_i-v) : amp
            I : amp '''
eqs_inh = '''dv/dt = (g_l*(v_r-v)+Ie+Ii+I)/(C_m) : volt
            dge/dt = -ge/(tau_exc) : siemens
            dgi/dt = -gi/(tau_inh) : siemens
            Ie = ge*(v_e-v) : amp
            Ii = gi*(v_i-v) : amp
            I : amp '''
eq_stdp = '''dx_post/dt = -x_post/tau_stdp : 1 (event-driven)
            dx_pre/dt = -x_pre/tau_stdp : 1 (event-driven)
            w: siemens '''
eq_pre = '''gi+=w
            w=clip(w+eta.eta*(x_post-alpha)*g_ei,g_min,g_max)
            x_pre+=1'''
eq_post = '''w=clip(w+eta.eta*x_pre*g_ei,g_min,g_max)
            x_post+=1'''


def if_else(condition, a, b) :
    if condition: return a
    else: return b


class Pointless(object):
    '''a hackaround changing learning rate eta'''
    pass
eta = Pointless()
eta.v = .001
eta.eta = 1.*eta.v

# defines an extra clock according to which some extra input currents 
# can be injected; 
# one can play with changing conductances etc...
"""
syn_input_freq=1.*Hz # frequency of current input oscillation
myclock = bb.Clock(dt=10*ms) # create an extra clock
@bb.network_operation(myclock)
def inject():
    '''
        Injects currents into neuronal populations...off by default
    '''
    if myclock.t>25000*ms:
        nn.Pe.I= nn.ext_input+\
                nn.Isine*(1.+0*np.sin(2*np.pi*myclock.t*syn_input_freq))
        nn.Pi.I= nn.ext_input+\
                nn.Isini*(1.+0*np.sin(2*np.pi*myclock.t*syn_input_freq))
"""

class Nets():
    def __init__(self, Ne=10000, Ni=2500, cp_ee=.02, cp_ie=.02, cp_ei=.02,
        cp_ii=.02, pr=.05, pf=.05, g_ee=0.19*nS, g_ie=0.2*nS, g_ei=1.0*nS,
        g_ii=1.0*nS, n_ass=10, s_ass=500, n_chains=0, cf_ffn=1., cf_rec=1.,
        type_ext_input='curr', ext_input=200*pA, synapses_per_nrn=250,
        inject_some_extra_i=False, g_ff_coef=1,
        symmetric_sequence=False, p_rev=0, extra_recorded_nrns=False,
        limit_syn_numbers=False, continuous_ass=False,
        use_random_conn_ff=False, modified_contin=False):
        '''
            Ne: number of excitatory neurons
            r_ie: ration of Ni/Ne
            cp_yx: connection probability from x to y
            if type_ext_input=='pois': ext_input={'N_p:10000','f_p':25,
                                        'coef_ep':1., 'sp':.02}
            !!!
            due to current limitations (that I wanna set all g_ee once and
            not to care of which how much it is), currently g_ff_coef can take
            only integer values, if I want a strong synapse, I just put several
            normal ones!
        '''
        ########################################################################
        # define a bunch of consts
        self.timestep = .1*ms       # simulation time step
        self.D = 2*ms               # AP delay
        self.m_ts = 1.*ms           # monitors time step
        
        if Ne>0:
            self.r_ie = (Ni+.0)/Ne            # ratio Ni/Ne
        else: 
            self.r_ie=.0
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne+self.Ni
        # set some random connectivity for all E,I neurons
        self.cp_ee = cp_ee
        self.cp_ie = cp_ie
        self.cp_ei = cp_ei
        self.cp_ii = cp_ii
        # conductances
        self.g_ee = g_ee             
        self.g_ie = g_ie              
        self.g_ei = g_ei              
        self.g_ii = g_ii              
        self.g_max = g_max              

        self.g_ff_coef = int(g_ff_coef) 
        self.g_l = g_l
        self.use_random_conn_ff = use_random_conn_ff 

        self.type_ext_input=type_ext_input
        self.ext_input=ext_input

        self.limit_syn_numbers = limit_syn_numbers
        self.n_chains = n_chains
        self.n_ass = n_ass   # number of assemblies in the ffn/minimum 2
        self.s_ass = s_ass  # neurons in an assembly
        self.s_assinh = int(self.s_ass*self.r_ie)

        self.cf_ffn = cf_ffn # strength of ffn synaptic connections
        self.cf_rec = cf_rec # strength of rec synaptic connections
        
        # recurrent connection probabilities into a group
        self.pr_ee = pr    # e to e
        self.pr_ie = pr    # e to i
        self.pr_ei = pr
        self.pr_ii = pr
        # FF connection probabilities
        self.pf_ee = pf
        self.pf_ie = 0#pf
        self.pf_ei = 0#pf
        self.pf_ii = 0#pf
        # FB maybe?
        self.symmetric_sequence= symmetric_sequence 
        self.continuous_ass = continuous_ass
        self.synapses_per_nrn = synapses_per_nrn 

        self.modified_contin = modified_contin 

        self.sh_e = 0
        self.sh_i = 0
        ########################################################################
        # neurons and groups to measure from
        self.nrn_meas_e =[] 
        self.nrn_meas_i = []
        # neuron groups for spike time measure (for cv and ff)
        if True:
            self.nrngrp_meas = [0, 5, self.n_ass-1]
            self.n_spikeM_gr = min(50, int(self.s_ass))
            
            # temporal recording from ps neurons
            self.nrn_meas_e.append(0*self.s_ass)
            self.nrn_meas_e.append(1*self.s_ass)
            self.nrn_meas_e.append(2*self.s_ass)
            self.nrn_meas_e.append(3*self.s_ass)
            self.nrn_meas_e.append((self.n_ass-1)*self.s_ass-1)
            self.nrn_meas_e.append((self.n_ass-1)*self.s_ass+1)
            self.nrn_meas_e.append((self.n_ass)*self.s_ass-1)

            # put a few neurons to measure for F2 plots
            for i in range(50):
                self.nrn_meas_e.append((self.n_ass)*self.s_ass-50-i)
            self.nrn_meas_i.append(1*self.s_assinh-1)

        self.nrn_meas_e.append(self.Ne-1)
        self.nrn_meas_i.append(self.Ni-1)

        if extra_recorded_nrns:
            # record extra all nrns in second, last assembly and random nrns
            for i in range(self.s_ass):
                self.nrn_meas_e.append(1*self.s_ass+i)
            for i in range(self.s_ass):
                self.nrn_meas_e.append((self.n_ass-1)*self.s_ass+i)
            for i in range(self.s_ass):
                self.nrn_meas_e.append(self.n_ass*self.s_ass+i)

        self.p_ass = []
        self.p_assinh = []
        self.p_ass_index = []
        self.p_assinh_index = []

        self.dummy_ass_index = [] # index of non-PS neurons, size is s_ass 
        # then function to apply them (later)

        self.dummy_group=[]
        self.C_ed = []

        self.inject_some_extra_i = inject_some_extra_i 
        self.p_rev = p_rev
        # define variables..needed??
        self.weights = []
        self.create_net()
        print 'inited ', asctime()
    
    def create_net(self):
        ''' create a network with and connect it'''
        self.network = bb.Network()
        self.network.clock = bb.Clock(dt=self.timestep)
        
        # create a couple of groups
        self.Pe = bb.NeuronGroup(self.Ne, eqs_exc, threshold=-50*mV,
                                            reset=-60*mV, refractory=2.*ms)
        self.Pi = bb.NeuronGroup(self.Ni, eqs_inh, threshold=-50*mV,
                                            reset=-60*mV, refractory=2.*ms)

        self.Pe.v = (-65 + 15*np.random.rand(self.Ne))*mV
        self.Pi.v = (-65 + 15*np.random.rand(self.Ni))*mV
        self.network.add(self.Pe, self.Pi) 
        if self.inject_some_extra_i:
            self.network.add(inject) 
       
        if self.type_ext_input=='curr':
            self.set_in_curr([self.Pe,self.Pi])
        elif self.type_ext_input=='pois':
            # apparently now works only with curr
            self.set_in_curr([self.Pe,self.Pi])
        else:
            print 'no input, sure about it?'

        self.C_ee=bb.Synapses(self.Pe,self.Pe,model='w:siemens',pre='ge+=w')
        self.C_ie=bb.Synapses(self.Pe,self.Pi,model='w:siemens',pre='ge+=w')
        self.C_ii=bb.Synapses(self.Pi,self.Pi,model='w:siemens',pre='gi+=w')
        stdp_on = True
        if stdp_on:
            namespace={'exp':np.exp,'clip':np.clip,'g_ei':self.g_ei}
            self.C_ei = bb.Synapses(self.Pi,self.Pe,
                model= eq_stdp, pre=eq_pre, post=eq_post,
                code_namespace=namespace)
        else:
            self.C_ei = bb.Synapses(self.Pi, self.Pe,
                    model='w:siemens', pre='gi+=w')

    def generate_ps_assemblies(self, ass_randomness='gen_no_overlap'):
        '''
            generates assemblies of random neurons,
            neurons can lie into several group, but once into the same group
            ass_randomness : how to pick the neurons
                    gen_ordered     : ordered assemblies
                    gen_no_overlap  : random assemblies, no overlap
                    gen_ass_overlap : random assemlies with overlap
                    gen_random      : totally random choise of neurons

        '''
        def gen_ordered(): 
            '''
                Generate n assemblies where neurons are ordered
                sh_e, sh_i : shift of e/i neurons (by default order starts at 0)
            '''
            if self.n_chains:
                self.sh_e += sa_e*self.n_ass 
                self.sh_i += sa_i*self.n_ass 
            nrn_e = np.arange(self.sh_e, self.Ne)
            nrn_i = np.arange(self.sh_i, self.Ni)
            p_ind_e= [nrn_e[n*sa_e:(n+1)*sa_e] for n in range(self.n_ass)]
            p_ind_i= [nrn_i[n*sa_i:(n+1)*sa_i] for n in range(self.n_ass)]
            print 'An ordered sequence is created'
            return p_ind_e, p_ind_i

        def gen_no_overlap(): 
            '''
                Generate n assemblies with random neurons
                no repetition of a neuron is allowed
            '''
            nrn_perm_e = np.random.permutation(self.Ne)
            nrn_perm_i = np.random.permutation(self.Ni)
            p_ind_e= [nrn_perm_e[n*sa_e:(n+1)*sa_e] for n in range(self.n_ass)]
            p_ind_i= [nrn_perm_i[n*sa_i:(n+1)*sa_i] for n in range(self.n_ass)]
            print 'A random sequence without overlaps is created'
            return p_ind_e, p_ind_i

        def gen_ass_overlap(): 
            '''
                Generate a n assemblies with random neurons
                repetitions of a neuron in different groups is allowed 
            '''
            # permuate and pick the first s_ass elements..
            p_ind_e = [np.random.permutation(self.Ne)[:sa_e] 
                    for n in range(self.n_ass)]
            p_ind_i = [np.random.permutation(self.Ni)[:sa_i] 
                    for n in range(self.n_ass)]
            print 'A random sequence without repetition in a group is created'
            return p_ind_e, p_ind_i

        def gen_random(): 
            '''
                Generate a n assemblies with random neurons, repetitions in a
                group are allowed
            '''
            p_ind_e = np.random.randint(self.Ne,size=(self.n_ass,sa_e))
            p_ind_i = np.random.randint(self.Ni,size=(self.n_ass,sa_i))
            print 'A sequence with completely random neurons is created'
            return p_ind_e, p_ind_i

        def gen_dummy():
            dum = []
            indexes_flatten = np.array(p_ind_e).flatten()
            # not to generate a random number for each neurons
            permutated_numbers = np.random.permutation(self.Ne) 
            dum_size= 0
            for nrn in  permutated_numbers:
                if nrn not in indexes_flatten:
                    dum.append(nrn)
                    dum_size+=1
                    if dum_size>=self.s_ass:
                        break
            return dum

        sa_e, sa_i = self.s_ass, self.s_assinh # to use shorter names
       
        p_ind_e, p_ind_i = eval(ass_randomness)()
        self.p_ass_index.append(p_ind_e)
        self.p_assinh_index.append(p_ind_i)
        self.dummy_ass_index.append(gen_dummy())
        self.n_chains += 1

    def set_net_connectivity(self):
        '''sets connections in the network'''
        def create_random_matrix(pre_nrns, post_nrns, p, pre_is_post=True):
            '''
                creates random connections between 2 populations of size
                pre_nrns and post_nrns (population sizes)
                might be slow but allows us to edit the connectivity matrix
                before throwing it into the ruthless synapse class
                ith element consists of the postsynaptic connection of ith nrn
                pre_is_post : flag that prevents a neuron to connect to itself
                    if set to True
            '''
            conn_mat = []
            for i in range(pre_nrns):
                conn_nrn = list(np.arange(post_nrns)\
                            [np.random.random(post_nrns)<p])
                if i in conn_nrn and pre_is_post: # no autosynapses
                    conn_nrn.remove(i)
                conn_mat.append(conn_nrn)
            return conn_mat

        def make_connections_discrete():
            for n_ch in range(self.n_chains):           # iterate over sequences
                p_index = self.p_ass_index[n_ch]
                p_indexinh = self.p_assinh_index[n_ch]
                # iterate over the assemblies in the PS
                for n_gr in range(len(p_indexinh)):
                    # iterate over E neurons in a group
                    for p1 in p_index[n_gr]:
                        # E to E recurrent
                        p1_post = list(p_index[n_gr][
                            np.random.random(len(p_index[n_gr]))<self.pr_ee])
                        if p1 in p1_post: # no autosynapse
                            p1_post.remove(p1)
                        if remove_old_conn_flag_ee:
                            cee[p1] = cee[p1][len(p1_post):]
                            if p1<5:
                                print n_gr, p1, len(p1_post)
                        cee[p1].extend(p1_post)
                        # E to E feedforward
                        if n_gr<self.n_ass-1: # in case it's the last group
                            ###################################################
                            # flag for using the random connections for ff
                            # instead of embedding new ff synapses, strengthen
                            # the background connections proportionally
                            use_random_conn_ff = False
                            if use_random_conn_ff:
                                p1_post = np.intersect1d(cee[p1],
                                                        p_index[n_gr+1]) 
                                for i in range(int(self.pf_ee/self.cp_ee)):
                                    cee[p1].extend(p1_post)
                                #check for postsynaptic partners of p1 in cee
                                # do the same synapses pff/r_rand times?
                                pass
                            else: 
                                for i in range(self.g_ff_coef):
                                    p1_post = list(p_index[n_gr+1]
                                        [np.random.random(len(p_index[n_gr+1]))
                                                                <self.pf_ee])
                                    if p1 in p1_post: # no autosynapse
                                        p1_post.remove(p1)
                                    if remove_old_conn_flag_ee:
                                        cee[p1] = cee[p1][len(p1_post):]
                                        if p1<5:
                                            print n_gr, p1, len(p1_post)
                                    cee[p1].extend(p1_post)
                        # E to E reverse
                        if self.symmetric_sequence:
                            if n_gr: # in case it's first group
                                p1_post = list(p_index[n_gr-1][
                                np.random.random(len(p_index[n_gr-1])) < \
                                    self.p_rev])
                                if p1 in p1_post: # no autosynapse
                                    p1_post.remove(p1)
                                if remove_old_conn_flag_ee:
                                    cee[p1] = cee[p1][len(p1_post):]
                                cee[p1].extend(p1_post)
                        # E to I recurrent
                        p1_post = list(p_indexinh[n_gr][
                            np.random.random(len(p_indexinh[n_gr]))<self.pr_ie])
                        if remove_old_conn_flag:
                            cie[p1] = cie[p1][len(p1_post):]
                        cie[p1].extend(p1_post)
                    #pr_ii = self.pr_ii/3
                    for i1 in p_indexinh[n_gr]:
                        # I to I recurrent
                        i1_post = list(p_indexinh[n_gr][
                            np.random.random(len(p_indexinh[n_gr]))<self.pr_ii])
                            #np.random.random(len(p_indexinh[n_gr]))<pr_ii])
                        if i1 in i1_post: # no autosynapse
                            i1_post.remove(i1)
                        if remove_old_conn_flag:
                            cii[i1] = cii[i1][len(i1_post):]
                        cii[i1].extend(i1_post)
                        '''
                        '''
                        # I to E recurrent
                        i1_post = list(p_index[n_gr][
                            np.random.random(len(p_index[n_gr]))<self.pr_ei])
                        if remove_old_conn_flag:
                            cei[i1] = cei[i1][len(i1_post):]
                        cei[i1].extend(i1_post)
            return cee, cie, cie, cii

        def make_connections_continuous():
            #def find_post(p_ind, i, hw, pr):
            def find_post(p_ind, i, ran_be, ran_af, pr):
                '''
                    hw stands for half width (M/2) normally 250 neurons
                    range variables specify the range of connectivity from
                    neuron i,i.e., to how many neurons will neuron i project
                        ran_be: range before neuron
                        ran_af: range after

                '''
                # rns from first group will have higher rc connection to 
                # the following half group
                if i < ran_be:
                    #pr_n = 2.*hw/(hw+i)*pr
                    pr_n = (ran_be+ran_af)/(ran_af+i)*pr
                    p1_post = p_ind[0:i+ran_af][\
                            np.random.random(i+ran_af)<pr_n]
                # last neurons also need some special care to connect
                elif i > len(p_ind) - ran_af:
                    #pr_n = 2.*hw/(hw+len(p_ind)-i-1)*pr
                    pr_n = pr*(ran_be+ran_af)/(ran_af+len(p_ind)-i-1)
                    p1_post = p_ind[i-ran_be:][\
                                np.random.random(len(p_ind)-i+ran_be)<pr_n]
                    print 'aa', len(p_ind), i, ran_be, ran_af, pr_n
                    print len(p_ind[i-ran_be:]), len(p_ind)-i+ran_be
                # most neurons are happy
                else:
                    pr_n = pr
                    p1_post = p_ind[i-ran_be:i+ran_af][
                                np.random.random(ran_be+ran_af)<pr_n]
                return p1_post

            for n_ch in range(self.n_chains):       # iterate over sequences
                p_index = np.array(self.p_ass_index[n_ch]).flatten()
                p_indexinh = np.array(self.p_assinh_index[n_ch]).flatten()
                ran_be = 1*self.s_ass/2 # here positive means before..to fix!
                ran_af = 1*self.s_ass/2
                ran_be_i = self.s_assinh/2+1
                ran_af_i = self.s_assinh/2+1
                if self.modified_contin:
                    ran_ff_start = 1*self.s_ass/2
                    ran_ff_end = 3*self.s_ass/2
                # iterate over the assemblies in the PS
                for i, p1 in enumerate(p_index):
                    # E-to-E recurrent
                    p1_post = find_post(p_index, i, ran_be, ran_af, self.pr_ee)
                    #if p1 in p1_post: # no autosynapse
                        #p1_post = list(p1_post).remove(p1)
                    cee[p1].extend(p1_post)

                    # E-to-I recurrent
                    p1_post = find_post(p_indexinh, i/4, ran_be_i, ran_af_i,
                                        self.pr_ie)
                    cie[p1].extend(p1_post)

                    # E-to-E feedforward
                    if i < len(p_index)-ran_ff_end:
                        p1_post = p_index[i+ran_ff_start:i+ran_ff_end][
                                    np.random.random(ran_ff_end-ran_ff_start)
                                    < self.pf_ee]
                    # here not to miss connections to the last group 
                    else:
                        p1_post = p_index[i:len(p_index)][
                                np.random.random(len(p_index)-i)<self.pf_ee]
                    cee[p1].extend(p1_post)

                for i, i1 in enumerate(p_indexinh):
                    # I-to-E recurrent
                    i1_post = find_post(p_index, 4*i,
                                        ran_be, ran_af, self.pr_ei)
                    cei[i1].extend(i1_post)

                    # I-to-I recurrent
                    i1_post = find_post(p_indexinh, i, ran_be_i, ran_af_i,
                                        self.pr_ii)
                    #if i1 in i1_post: # no autosynapse
                        #i1_post = list(i1_post).remove(i1)
                    cii[i1].extend(i1_post)

        def apply_connection_matrix(S, conn_mat, f_ee=False):
            '''
                creates the synapses by applying conn_mat connectivity matrix
                to the synaptic class S
                basically does the following but fast!

                for i, conn_nrn in enumerate(conn_mat):
                    for j in conn_nrn:
                        S[i,j]=True

                f_ee is a flag indicating e-e connections

            '''
            presynaptic, postsynaptic = [], []
            synapses_pre = {}
            nsynapses = 0
            for i in range(len(conn_mat)):
                conn_nrn = conn_mat[i]
                k1 = len(conn_nrn)
                # too connected? get rid of older synapses
                if self.limit_syn_numbers and f_ee and (k1>self.synapses_per_nrn): 
                    #conn_nrn = conn_nrn[self.synapses_per_nrn:] # simply cut!
                    x = max(self.synapses_per_nrn, k1-self.synapses_per_nrn)
                    conn_nrn = conn_nrn[-x:] # simply cut!
                    '''
                    # some exponential forgeting of old synapses
                    tau = (k1-self.synapses_per_nrn)/2.
                    conn_nrn = np.array(conn_nrn)[\
                        np.exp(-np.arange(k1)/tau)<np.random.random(k1)]
                    '''
                k = len(conn_nrn) # new number of postsynaptic connections
                # just print to keep an eye on what's going on
                #if i<20:
                    #print '# synpapses before and after ', k1,k
                if k:
                    synapses_pre[i] = nsynapses + np.arange(k)
                    presynaptic.append(i*np.ones(k, dtype=int))
                    postsynaptic.append(conn_nrn)
                    nsynapses += k
            presynaptic = np.hstack(presynaptic)
            postsynaptic = np.hstack(postsynaptic)
            S.create_synapses(presynaptic, postsynaptic, synapses_pre)

        # creates randomly connected matrices
        cee = create_random_matrix(self.Ne, self.Ne, self.cp_ee, True)
        cie = create_random_matrix(self.Ne, self.Ni, self.cp_ie, False)
        cei = create_random_matrix(self.Ni, self.Ne, self.cp_ei, False)
        cii = create_random_matrix(self.Ni, self.Ni, self.cp_ii, True)

        # seems that these 2 flags are outdated and unusable; can't bother to
        # remove them now
        remove_old_conn_flag_ee = False
        remove_old_conn_flag = False
        ########################################################################
        ### now imprint PS
        ########################################################################
        if self.continuous_ass:
            make_connections_continuous()
        else:
            make_connections_discrete()    

        apply_connection_matrix(self.C_ee, cee, True)
        apply_connection_matrix(self.C_ie, cie)
        apply_connection_matrix(self.C_ei, cei)
        apply_connection_matrix(self.C_ii, cii)

        self.C_ee.w = self.g_ee
        self.C_ie.w = self.g_ie
        self.C_ei.w = self.g_ei
        self.C_ii.w = self.g_ii
        self.C_ee.delay = self.D
        self.C_ie.delay = self.D
        self.C_ei.delay = self.D
        self.C_ii.delay = self.D
        self.network.add(self.C_ee)
        self.network.add(self.C_ie)
        self.network.add(self.C_ei)
        self.network.add(self.C_ii)

        self.weights.append(self.C_ei.w.data.copy()) #save weights
        print 'connections imprinted! ', asctime()

    def boost_pff(self, pf_ee_new):
        '''
            creates anew connectivity matrix and applies to code
            for new ff connections that should be added after some
            simulation time

        '''
        def get_disc_conn():
            conn_mat = [[] for i in range(self.Ne)]
            # E to E feedforward
            for ch in range(self.n_chains):
                p_index = self.p_ass_index[ch]
                for gr in range(self.n_ass-1):
                    for p1 in p_index[gr]:
                        p1_post = list(p_index[gr+1]
                            [np.random.random(len(p_index[gr+1])) \
                                    < self.pf_ee_new])
                        conn_mat[p1].extend(p1_post)
            return conn_mat
     
        def get_cont_conn():
            conn_mat = [[] for i in range(self.Ne)]
            if self.modified_contin:
                ran_ff_start = 1*self.s_ass/2
                ran_ff_end = 3*self.s_ass/2
            for ch in range(self.n_chains):
                p_index = np.array(self.p_ass_index[ch]).flatten()
                for i, p1 in enumerate(p_index):
                    # E-to-E feedforward
                    if self.modified_contin:
                        if i < len(p_index)-ran_ff_end:
                            p1_post = p_index[i+ran_ff_start:i+ran_ff_end][
                                np.random.random(ran_ff_end-ran_ff_start)
                                <self.pf_ee_new]
                        # here not to miss connections to the last group 
                        elif i < len(p_index)-ran_ff_start:
                            p1_post = p_index[i+ran_ff_start:len(p_index)][
                                np.random.random(len(p_index)-i-ran_ff_start)
                                <self.pf_ee_new]
                        else:
                            p1_post=[]
                    else:
                        if i < len(p_index)-self.s_ass:
                            p1_post = p_index[i:i+self.s_ass][
                                np.random.random(self.s_ass)<self.pf_ee_new]
                        # here not to miss connections to the last group 
                        else:
                            p1_post = p_index[i:len(p_index)][
                                np.random.random(len(p_index)-i)<self.pf_ee_new]
                    conn_mat[p1].extend(p1_post)
            return conn_mat 

        def get_rand_boost():
            ex_pre = np.array(self.C_ee.presynaptic)
            ex_post = np.array(self.C_ee.postsynaptic)

            conn_mat = [[] for i in range(self.Ne)]
            for ch in range(self.n_chains):
                p_index = self.p_ass_index[ch]
                for gr in range(self.n_ass-1):
                    for p1 in p_index[gr]:
                        p1_ex_post = ex_post[ex_pre==p1]
                        p1_post = np.intersect1d(
                                    self.p_ass_index[0][gr+1], p1_ex_post)
                        for i in range(int(self.pf_ee_new/self.cp_ee)):
                            conn_mat[p1].extend(p1_post)
                        if not gr and not p1:
                            print p1, p1_post
                            print 
            #1/0
            return conn_mat

        self.pf_ee_new = pf_ee_new
        self.C_ee_ff = bb.Synapses(self.Pe, self.Pe,
                        model='w:siemens', pre='ge+=w')
        if self.continuous_ass:
            conn_mat = get_cont_conn()
        else:
            if self.use_random_conn_ff:
                conn_mat = get_rand_boost()
            else:
                conn_mat = get_disc_conn()

        presynaptic, postsynaptic = [], []
        synapses_pre = {}
        nsynapses = 0
        for i in range(len(conn_mat)):
            conn_nrn = conn_mat[i]
            k = len(conn_nrn) # new number of postsynaptic connections
            if k:
                synapses_pre[i] = nsynapses + np.arange(k)
                presynaptic.append(i*np.ones(k, dtype=int))
                postsynaptic.append(conn_nrn)
                nsynapses += k
        presynaptic = np.hstack(presynaptic)
        postsynaptic = np.hstack(postsynaptic)

        self.C_ee_ff.create_synapses(presynaptic, postsynaptic, synapses_pre)
        self.C_ee_ff.w = self.g_ee
        self.C_ee_ff.delay = self.D
        self.network.add(self.C_ee_ff)
        print 'pff boosted!'

    def balance(self, bal_time=2*second, eta_c=1.):
        """
        balancing function: runs the network for bal_time and:
        1) sets the learning rate to eta
        2) !!! switches off the spike recorder (ap_record = False)

        """
        t0 = time()
        eta.eta = eta.v*eta_c
        self.network.run(bal_time)
        # save weights after each balance
        self.weights.append(self.C_ei.w.data.copy())         
        eta.eta = 0.0
        t1 = time()
        print 'balanced: ', t1-t0

    def run_sim(self, run_time= 1*second):
        """ runs the network for run_time with I plasticity turned off"""
        t0 = time()
        eta.eta = 0.0
        self.network.run(run_time)
        t1 = time()
        print 'run: ', t1-t0
        
    def set_in_curr(self, target, ext_input=None):
        """ ce,ci currents injected in E/I populations"""
        if ext_input==None:
            ext_input = self.ext_input
        for t in target:
            t.I = ext_input
    
    def set_in_poisson(self, target):
        """
            Set poissonian input to a group of neurons
            target: list of targert groups
            N_p: # of poissons inputs
            f_p: frequency of P
            sp: sparseness of connections
            coef_ep: factor of ep conductance to g_exc

        """
        ## somehow PoissonInput is way slower! also leads to diff behaviour
        #for gr in target:
            #inp_poisson = bb.PoissonInput(gr,N=100,rate=f_p,
                #weight=2.*self.g_ee,state='ge')
            #self.network.add(inp_poisson)
        N_p=self.ext_input['N_p']
        f_p=self.ext_input['f_p']
        sp=self.ext_input['sp']
        coef_ep=self.ext_input['coef_ep']
        self.P_poisson= bb.PoissonGroup(N_p,f_p,self.network.clock)
        self.network.add(self.P_poisson)
        for gr in target:
            #Cep = bb.Connection(self.P_poisson, gr,'ge', 
            #                        weight=coef_ep*self.g_ee, sparseness=sp)
            Cep= bb.Synapses(self.P_poisson,gr,model='w:siemens',pre='ge+=w')
            Cep.connect_random(self.P_poisson,gr,sparseness=sp)
            Cep.w=coef_ep*self.g_ee
            self.network.add(Cep)

    def set_syn_input(self, target, time):
        '''adding sync inputs at some time points''' 
        ext_in = bb.SpikeGeneratorGroup(1,[(0, time)],self.network.clock)
        C_syne= bb.Synapses(ext_in,target,model='w:siemens',pre='ge+=w')
        C_syne.connect_random(ext_in,target,sparseness=1.)
        C_syne.w=30.*self.g_ee
        self.network.add(ext_in, C_syne)
    
    def set_syn_input_ran(self, target, time):
        '''adding sync inputs at some time points''' 
        ext_in = bb.SpikeGeneratorGroup(1,[(0, time)],self.network.clock)
        C_syne= bb.Synapses(ext_in,self.Pe,model='w:siemens',pre='ge+=w')
        for n in target:
            C_syne.connect_random(ext_in,self.Pe[n],sparseness=1.)
        C_syne.w=30.*self.g_ee
        self.network.add(ext_in, C_syne)
    
    def set_noisy_input(self, target, time, sigma=0., mcoef=30):
        '''adding sync inputs at some time points with 
            normal jitter distribution sigma

            mcoef is the strength of stimulation
            
        ''' 
        #print time, sigma
        t0 = time - 6.*sigma # mean delay is set to 6*sigma
        ext_in = bb.SpikeGeneratorGroup(1, [(0, t0)], self.network.clock)
        C_syne = bb.Synapses(ext_in, self.Pe, model='w:siemens', pre='ge+=w')
        for n in target:
            C_syne.connect_random(ext_in, self.Pe[n], sparseness=1.)
        C_syne.w = mcoef * self.g_ee
        #C_syne.delay=np.random.uniform(0,sigma,len(target))
        if sigma > 0.:
            C_syne.delay = np.random.normal(6.*sigma, sigma, len(target))
        else:
            C_syne.delay = np.zeros(len(target))
        self.network.add(ext_in, C_syne)

    def attach_dummy_group(self, pf=.06):
        self.dummy_group = bb.NeuronGroup(500, eqs_exc, threshold=-50*mV,
                                            reset=-60*mV, refractory=2.*ms)
        self.C_ed=bb.Synapses(self.dummy_group,self.Pe,
                                        model='w:siemens',pre='ge+=w')
        for p1 in self.dummy_group:
            for p2 in p_index[n_gr+1]:  
                if np.random.random()<nn.pf_ee:
                    self.C_ed[p1,p2]=True
                    #self.C_ee[p1,p2].w=self.g_ee

        print 'hui'
        #nn.C_ed.connect_random(nn.dummy_group,nn.p_ass_index[0][0],sparseness=pf)
        self.C_ed.w=self.g_ee
        self.C_ed.delay=self.D
        self.network.add(self.dummy_group,self.C_ed)

    def set_rate_monitor(self):
        """yep"""
        self.mon_rate_e = bb.PopulationRateMonitor(self.Pe, bin = self.m_ts)
        self.mon_rate_i = bb.PopulationRateMonitor(self.Pi, bin = self.m_ts)
        self.network.add(self.mon_rate_e, self.mon_rate_i)
    
    def set_spike_monitor(self):
        """yep"""
        self.mon_spike_e = bb.SpikeMonitor(self.Pe)
        self.mon_spike_i = bb.SpikeMonitor(self.Pi)
        self.network.add(self.mon_spike_e, self.mon_spike_i)
    
    def set_group_spike_monitor(self, ch=0):
        """
            !!!
            this would not work with random assemblies
            to be removed in the future
        """
        self.mon_spike_sngl = [] # measure spike times from a few single neurons
        for nrn in self.nrn_meas_e:
            self.mon_spike_sngl.append(bb.SpikeMonitor(self.Pe[nrn]))
        self.network.add(self.mon_spike_sngl)

        self.mon_spike_gr = [] # measure spike times from groups (for CV and FF)
        for gr in self.nrngrp_meas:
            self.mon_spike_gr.append(bb.SpikeMonitor(
                                        self.p_ass[ch][gr][0:self.n_spikeM_gr]))
        # also control group of neurons which is not included in the ps
        self.mon_spike_gr.append(bb.SpikeMonitor(\
                    self.Pe[self.n_ass*self.s_ass:(self.n_ass+1)*self.s_ass]
                    [0:self.n_spikeM_gr]))
        self.network.add(self.mon_spike_gr)
        # default spike easure is off
        for sp in self.mon_spike_gr:
            sp.record = False 
        
    def set_voltage_monitor(self):
        """yep"""
        self.mon_volt_e = bb.StateMonitor(self.Pe, 'v', record=self.nrn_meas_e)
        self.mon_volt_i = bb.StateMonitor(self.Pi, 'v', record=self.nrn_meas_i)
        self.network.add(self.mon_volt_e ,self.mon_volt_i)
    
    def set_conductance_monitor(self):
        """yep"""
        self.mon_econd_e= bb.StateMonitor(self.Pe,'ge',record=self.nrn_meas_e)
        self.mon_icond_e= bb.StateMonitor(self.Pe,'gi',record=self.nrn_meas_e)
        self.mon_econd_i= bb.StateMonitor(self.Pi,'ge',record=self.nrn_meas_i)
        self.mon_icond_i= bb.StateMonitor(self.Pi,'gi',record=self.nrn_meas_i)
        self.network.add(self.mon_econd_e, self.mon_icond_e,
                        self.mon_econd_i ,self.mon_icond_i)

    def set_current_monitor(self):
        """yep"""
        self.mon_ecurr_e= bb.StateMonitor(self.Pe, 'Ie', record=self.nrn_meas_e)
        self.mon_icurr_e= bb.StateMonitor(self.Pe, 'Ii', record=self.nrn_meas_e)
        self.mon_ecurr_i= bb.StateMonitor(self.Pi, 'Ie', record=self.nrn_meas_i)
        self.mon_icurr_i= bb.StateMonitor(self.Pi, 'Ii', record=self.nrn_meas_i)
        self.network.add(self.mon_ecurr_e ,self.mon_icurr_e,
                self.mon_ecurr_i,self.mon_icurr_i)

    def run_full_sim(self, sim_times):
        self.generate_ordered_ps()
        self.set_ffchain_new()
        
        self.set_rate_monitor()
        self.set_group_spike_monitor()
        #self.set_voltage_monitor()
        #self.set_current_monitor()

        stim_times=np.arange(sim_times['start_sim'],sim_times['stop_sim'],1)
        for t in stim_times:
            self.set_syn_input(self.p_ass[0][0],t*second)
            
        # stimulation with a que (not full)
        for que in [80,60,40,20]:
            start_que = sim_times['start_sim'+str(que)]
            stop_que = sim_times['stop_sim'+str(que)]
            que_res = que/100. #      # 80,60,40,20% of pop stimulation 
            
            for t in range(start_que, stop_que):
                n_sim_nrn = int(que_res*self.s_ass)
                self.set_syn_input(self.p_ass[0][0][0:n_sim_nrn],t*second)
      
        # set balance times with corresponding learning rates
        t0=0
        for t,r in zip(sim_times['balance_dur'],sim_times['balance_rate']):
            self.balance((t-t0)*second,r)
            t0=t
        
        # run the simulations
        self.run_sim((sim_times['stop_sim20']-sim_times['start_sim'])*second)
        # turn on the group spike monitor
        for sp in self.mon_spike_gr:
            sp.record = True 
        # run for spontan activity 
        self.run_sim((sim_times['stop_spont_recording']-
                                            sim_times['stop_sim20'])*second)

    def dummy(self):
        sim_times={}
        #sim_times['balance_dur']=[10,20,25,35]
        sim_times['balance_dur']=[10,15,20,25]
        sim_times['balance_rate']=[5,1,.1,.01]
        sim_times['start_sim']=16
        sim_times['stop_sim']=20
        sim_times['start_sim80']=20
        sim_times['stop_sim80']=20
        sim_times['start_sim60']=20
        sim_times['stop_sim60']=20
        sim_times['start_sim40']=20
        sim_times['stop_sim40']=20
        sim_times['start_sim20']=20
        sim_times['stop_sim20']=22
        sim_times['start_fr_recording']=16
        sim_times['stop_fr_recording']=25
        sim_times['start_spont_recording']=sim_times['stop_sim20']
        sim_times['stop_spont_recording']=25

        self.set_rate_monitor()
        self.set_spike_monitor()
        self.set_voltage_monitor()
        self.set_current_monitor()
        self.set_conductance_monitor()

        self.run_full_sim(sim_times)
    
    def plot_for_raster_curr_volt(self):
        num_ps = 1
        for n in range(num_ps):
            self.generate_ps_assemblies('gen_ass_overlap')
        self.set_net_connectivity()
        
        self.set_spike_monitor()
        self.set_rate_monitor()

        '''
        gr = self.p_ass_index[0][0]
        self.set_noisy_input(gr,.5*second,sigma=0*ms)
        #gr1 = self.p_ass_index[1][0]
        #self.set_noisy_input(gr1,.7*second,sigma=0*ms)
        self.balance(1.*second,5.)
        '''

        t0 = 30 # time offset for stimulation in secs
        n_stim = 5
        for n in range(num_ps):
            for i in range(n_stim):
                #gr = self.p_ass_index[n][0]
                gr_num = int(self.n_ass/5.*i)
                print 'stim to ', gr_num
                gr = self.p_ass_index[n][gr_num]
                t = (t0 + n + i*3)*second
                self.set_noisy_input(gr,t,sigma=0*ms)

        self.balance(10*second, 5.)
        self.balance(10*second, 1.)
        self.balance(5*second, .1)
        self.balance(5*second, .01)
        #self.run_sim((2*num_ps+2)*second)
        self.run_sim(16*second)

        for n in range(num_ps):
            figure = plt.figure(figsize=(12., 8.))
            plotter.plot_ps_raster(self, chain_n=n, frac=.01, figure=figure)
        #plotter.plot_pop_raster(self,False)
        #plt.xlim([19000,22000])
        #plt.savefig('xxx'+ str(pr)+'_'+str(pf)+'.png')
        #plotter.show()

    def test_shifts(self, ie, ii, tr):
        self.generate_ps_assemblies('gen_no_overlap')
        self.set_net_connectivity()
        self.set_spike_monitor()
        self.set_rate_monitor()

        #ie, ii = 0,3
        self.Isine=ie*pA
        self.Isini=ii*pA
        self.network.add(inject) 

        gr = self.p_ass_index[0][0]
        for i in range(9):
            t = (21+i)*second
            self.set_noisy_input(gr,t,sigma=0*ms)

        '''
        self.balance(.1*second,5.)
        '''
        self.balance(5*second,5.)
        self.balance(5*second,1.)
        self.balance(5*second,.1)
        self.balance(5*second,.01)
        self.run_sim(10*second)

        pr, pf = self.pr_ee, self.pf_ee
        
        figure = plt.figure(figsize=(12.,8.))
        plotter.plot_ps_raster(self,chain_n=0,frac=.1,figure=figure)
        #plt.xlim([6800,8300])
        '''
        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        '''

        # save the spikes data into a file fr later reading
        spikes_e = [self.mon_spike_e[nrn][self.mon_spike_e[nrn] > 20.]
                    for gr in self.p_ass_index[0] for nrn in gr]  

        #fname = '../data/dynamic_switch/pr05pf05aie1ii-1pA_20_25_30sec'
        #np.savez(fname,self.p_ass_index, spikes_e)

        #prefix = '../data/evoked_replay_extra_currents/'
        #fname = str(pr)+'_'+str(pf)+'ie'+str(ie)+'ii'+str(ii)+'_'+str(tr)
        #np.savez(prefix+fname,self.p_ass_index, spikes_e)
        '''
        plt.savefig(prefix+fname+'.png')
        plt.savefig(prefix+fname+'.pdf')
        '''
 
    def stim_curr(self, ps=0, gr=0, dur_stim=100, dur_relx=400,
                    curr=10*pA):
        '''
            stimulate group gr in ps with a continuous current

        '''
        for nrn in self.p_ass_index[ps][gr]:
            self.Pe[nrn].I += curr
        self.run_sim(dur_stim*ms)
        for nrn in self.p_ass_index[ps][gr]:
            self.Pe[nrn].I -= curr
        self.run_sim(dur_relx*ms)


def test_symm():

    from matplotlib import pyplot as plt
    nn=Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
        n_ass=10, s_ass=500, pr=.15, pf=.03, symmetric_sequence=True, p_rev=.03,
        g_ee=0.1*nS, g_ie=0.1*nS, g_ei=0.4*nS, g_ii=0.4*nS)

    nn.generate_ps_assemblies('gen_no_overlap')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()

    '''
    gr = nn.p_ass_index[0][0]
    t = 20*second
    nn.set_noisy_input(gr,t,sigma=0*ms)
    t = 20.5*second
    nn.set_noisy_input(gr,t,sigma=0*ms)

    gr = nn.p_ass_index[0][9]
    t = 21*second
    nn.set_noisy_input(gr,t,sigma=0*ms)
    t = 21.5*second
    nn.set_noisy_input(gr,t,sigma=0*ms)

    nn.balance(5*second,5.)
    nn.balance(5*second,1.)
    nn.balance(5*second,.1)
    nn.balance(5*second,.01)
    nn.run_sim(2*second)
    '''

    #gr = nn.p_ass_index[0][0]
    #t = 20.5*second
    #nn.set_noisy_input(gr,t,sigma=0*ms)

    for gr_num in range(nn.n_ass):
        gr = nn.p_ass_index[0][gr_num]
        t = (20.55+gr_num*.1)*second
        nn.set_noisy_input(gr,t,sigma=0*ms)

    #gr = nn.p_ass_index[0][9]
    #t = 22.5*second
    #nn.set_noisy_input(gr,t,sigma=0*ms)

    nn.balance(5*second,5.)
    nn.balance(5*second,1.)
    nn.balance(5*second,.1)
    nn.balance(5*second,.01)
    #nn.run_sim(4*second)
    nn.Pe.I -= .0*pA

    for nrn in nn.p_ass_index[0][0]:
        nn.Pe[nrn].I += 3*pA
    nn.run_sim(.5*second)
    for nrn in nn.p_ass_index[0][0]:
        nn.Pe[nrn].I -= 3*pA

    nn.Pe.I -= 9*pA
    nn.run_sim(1.*second)
    nn.Pe.I += 9*pA

    for nrn in nn.p_ass_index[0][9]:
        nn.Pe[nrn].I += 3*pA
    nn.run_sim(.5*second)
    for nrn in nn.p_ass_index[0][9]:
        nn.Pe[nrn].I -= 3*pA
    
    #nn.Pe.I +=.5*pA
    #nn.Pe.I +=5*pA
    #nn.run_sim(1.*second)

    #for nrn in nn.p_ass_index[0][0]:
        #nn.Pe[nrn].I += 1*pA
    #nn.run_sim(1.*second)

    plotter.plot_ps_raster(nn, chain_n=0, frac=.1)
    plt.xlim([20000, 22000])
    return nn


def test_fr():
    from matplotlib import pyplot as plt

    pr, pf = 0.06, 0.06

    nn = Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
        n_ass=10, s_ass=500, pr=pr, pf=pf,
        g_ee=0.1*nS, g_ie=0.1*nS, g_ei=0.4*nS, g_ii=0.4*nS)

    nn.generate_ps_assemblies('gen_no_overlap')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()

    gr = nn.p_ass_index[0][0]
    t = 20*second
    nn.set_noisy_input(gr,t,sigma=0*ms)
    t = 21*second
    nn.set_noisy_input(gr,t,sigma=0*ms)

    '''
    nn.balance(.01*second,5.)
    nn.balance(.01*second,1.)
    '''

    nn.balance(1*second,5.)
    #nn.balance(1*second,1.)
    #nn.balance(1*second,.1)
    #nn.balance(1*second,.01)
    #nn.run_sim(1*second)

    gr_fr_e = calc_spikes.make_fr_from_spikes(nn,ps=0,w=1,exc_nrns=True)
    gr_fr_i = calc_spikes.make_fr_from_spikes(nn,ps=0,w=1,exc_nrns=False)

    plt.subplot(211)
    for gr in range(nn.n_ass):
        plt.plot(calc_spikes.gaus_smooth(gr_fr_e[gr],2))
    plt.subplot(212)
    for gr in range(nn.n_ass):
        plt.plot(calc_spikes.gaus_smooth(gr_fr_i[gr],2))

    plt.show()

    return nn


def test_noPS():
    from matplotlib import pyplot as plt
    pr, pf = 0.06, 0.06
    nn=Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
        n_chains=0, n_ass=2, s_ass=500, pr=pr, pf=pf,
        g_ee=0.1*nS, g_ie=0.1*nS, g_ei=0.4*nS, g_ii=0.4*nS)

    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()

    nn.balance(5*second,5.)
    nn.balance(5*second,1.)
    nn.balance(5*second,.2)
    nn.balance(5*second,.05)
    nn.run_sim(1*second)
    return nn


def test_diff_gff(Ne=20000):
    gfc = 1
    pr = 0.06
    #pf= 0.06/gfc
    pf = 0.06

    #Ne=20000
    Ni = Ne/4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1*nS
    gi0 = 0.4*nS
 
    #gee= ge0
    gee = ge0*(20000./Ne)**.5
    gii = gi0*(20000./Ne)**.5


    pf = pf*(Ne/20000.)**.5
    pr = pr*(Ne/20000.)**.5

    # so that gfc*gee=g0 or the ff connection dont scale down
    #gfc = 1./(20000./Ne)**.5 
    #grc = 1./(20000./Ne)**.5 # figure out this guy

    continuous_ass=False
    nn = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
        n_ass=10, s_ass=500, pr=pr, pf=pf, ext_input=200*pA,
        g_ee=gee, g_ie=gee, g_ei=gii, g_ii=gii, g_ff_coef=gfc,
        continuous_ass=continuous_ass)

    #nn.generate_ps_assemblies('gen_no_overlap')
    nn.generate_ps_assemblies('gen_ordered')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()

    #return nn

    gr = nn.p_ass_index[0][0]
    '''
    '''
    #t = 20*second
    #nn.set_noisy_input(gr, t, sigma=0*ms) 
    t = 21*second
    nn.set_noisy_input(gr, t, sigma=0*ms) 
    #t = 22*second
    #nn.set_noisy_input(gr, t, sigma=0*ms)
    t = 23*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    #t = 24*second
    #nn.set_noisy_input(gr, t, sigma=0*ms)

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 

    '''
    nn.balance(20*second,5.)
    nn.balance(20*second,1.)
    nn.balance(10*second,.1)
    nn.balance(10*second,.01)
    '''
    #nn.balance(.5*second, 5.)
    #return nn

    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(5*second, .01)
    nn.mon_spike_e.record = True 
    nn.mon_spike_i.record = True 
    nn.run_sim(5*second)

    '''
    t_end = nn.network.clock.t*1000/second
    figure = plt.figure(figsize=(16., 12.))
    plotter.plot_ps_raster(nn, chain_n=0, frac=1., figure=figure,
                            dummy_ass=True)
    plt.xlim([t_end-2000, t_end])
    '''
    #fname_pre = 'large_nets/scaleAllg_Ne'
    #plt.savefig(fname_pre + str(Ne) + 'pr' + str(pr) + 'pf' + str(pf) + 

    #figure2 = plt.figure(figsize=(16.,12.))
    #plt.plot(nn.mon_rate_e.smooth_rate(5*ms))
    #plt.xlim([5000,t_end])
    #plt.xlabel('t [ms]')
    #plt.ylabel('FR [sp/sec]')
    #plt.savefig(fname_pre+ str(Ne) + 'pr'+str(pr)+'pf'+str(pf) + '_fr'+'.png')

    #plt.close('all')
    #plotter.plot_fr_cv_syn_distr(nn)
    return nn


def test_psps():
    '''
        test PSPs
    '''
    from matplotlib import pyplot as plt
    ge0 = 0.1*nS
    gi0 = 0.4*nS
    cp = 0
    nn=Nets(Ne=10,Ni=2,cp_ee=cp,cp_ie=cp,cp_ei=cp,cp_ii=cp,
        n_ass=0,s_ass=1,pr=0,pf=0,ext_input=0*pA,
        g_ee=ge0,g_ie=ge0,g_ei=gi0,g_ii=gi0)

    nn.C_ee=bb.Synapses(nn.Pe,nn.Pe,model='w:siemens',pre='ge+=w')
    nn.C_ee[0,9] = True
    nn.C_ee.w=nn.g_ee
    nn.C_ee.delay=nn.D
    nn.network.add(nn.C_ee)

    '''
    '''
    target = nn.Pe[0]
    ext_in = bb.SpikeGeneratorGroup(1,[(0, 300*ms)],nn.network.clock)
    C_syne= bb.Synapses(ext_in,target,model='w:siemens',pre='ge+=w')
    C_syne.connect_random(ext_in,target,sparseness=1.)
    C_syne.w = 130.*nn.g_ee
    nn.network.add(ext_in, C_syne)
    #nn.nrn_meas_e = nn.Pe
    nn.nrn_meas_e = [0,1,9]
    nn.mon_volt_e = bb.StateMonitor(nn.Pe, 'v', record=nn.nrn_meas_e)#,timestep=1)
    nn.network.add(nn.mon_volt_e)

    nn.run_sim(500*ms)

    plt.plot(nn.mon_volt_e.times/ms,
            nn.mon_volt_e[9]/mV)
    plotter.show()

    return nn


def test_longseq():

    nn=Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
        n_ass=444, s_ass=150, pr=.19, pf=.19, synapses_per_nrn=200,
        ext_input=200*pA, limit_syn_numbers=True,
        g_ee=0.1*nS, g_ie=0.1*nS, g_ei=0.4*nS, g_ii=0.4*nS
        )
    nn.generate_ps_assemblies('gen_ass_overlap')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()

    gr = nn.p_ass_index[0][0]
    t = 21*second
    nn.set_noisy_input(gr, t, sigma=0*ms, mcoef=30)

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 
    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(5*second, .01)
    nn.mon_spike_e.record = True
    nn.mon_spike_i.record = True
    nn.run_sim(6*second)

    #plotter.plot_ps_raster(nn, frac=1./150)

    fname = 'longseq444.npz'
    spikes4save = calc_spikes.get_spike_times_ps(nn, frac=1./150)
    np.savez_compressed(fname, spikes4save)

    return nn


def test_2ASS(Ne=20000, nass=2):
    gfc = 1
    pr = 0.1
    #pf= 0.06/gfc
    pf = 0.06

    #Ne=20000
    Ni = Ne/4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1*nS
    gi0 = 0.4*nS
 
    #gee= ge0
    # so that gfc*gee=g0 or the ff connection dont scale down
    #gfc = 1./(20000./Ne)**.5 
    #grc = 1./(20000./Ne)**.5 # figure out this guy

    nn = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
        n_ass=10, s_ass=500, pr=pr, pf=pf, ext_input=200*pA,
        g_ee=ge0, g_ie=ge0, g_ei=gi0, g_ii=gi0)

    #nn.generate_ps_assemblies('gen_no_overlap')
    #nn.generate_ps_assemblies('gen_no_overlap')
    nn.generate_ps_assemblies('gen_ordered')
    nn.generate_ps_assemblies('gen_ordered')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()

    #return nn

    gr0 = nn.p_ass_index[0][0]
    gr1 = nn.p_ass_index[1][0]
    '''
    '''
    #t = 20*second
    #nn.set_noisy_input(gr, t, sigma=0*ms) 
    t = 21*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22*second
    nn.set_noisy_input(gr1, t, sigma=0*ms)
    #t = 23*second
    #nn.set_noisy_input(gr, t, sigma=0*ms)
    #t = 24*second
    #nn.set_noisy_input(gr, t, sigma=0*ms)

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 

    '''
    nn.balance(20*second,5.)
    nn.balance(20*second,1.)
    nn.balance(10*second,.1)
    nn.balance(10*second,.01)
    '''
    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(5*second, .01)
    nn.mon_spike_e.record = True 
    nn.mon_spike_i.record = True 
    nn.run_sim(5*second)

    #figure = plt.figure(figsize=(16., 12.))
    #plotter.plot_pop_raster(nn)
    #plt.xlim([t_end-2000, t_end])
    #fname_pre = '2Ass'
    #plt.savefig(fname_pre+ str(Ne) + 'pr'+str(pr)+'pf'+str(pf) + '_fr'+'.png')

    fname = '2asss.npz'
    spikes4save = calc_spikes.get_all_spikes(nn)
    np.savez_compressed(fname, np.array(spikes4save))
    return nn


def show_ass_frs():
    '''
        Plots the firing of sequent assemblies

    '''

    nn=Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
        n_ass=10, s_ass=500, pr=.06, pf=.06,
        ext_input=200*pA,
        g_ee=0.1*nS, g_ie=0.1*nS, g_ei=0.4*nS, g_ii=0.4*nS
        )
    nn.generate_ps_assemblies('gen_ordered')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()
    nn.set_conductance_monitor()

    gr0 = nn.p_ass_index[0][0]
    t = 21*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 21.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 
    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(5*second, .01)
    nn.mon_spike_e.record = True
    nn.mon_spike_i.record = True
    nn.run_sim(6*second)
    plotter.plot_gr_fr2(nn, wbin=.2, ngroups=8)

    return nn


def test_tau():
    nn=Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
        n_ass=1, s_ass=500, pr=.00, pf=.00,
        ext_input=200*pA,
        g_ee=0.1*nS, g_ie=0.1*nS, g_ei=0.4*nS, g_ii=0.4*nS
        )
    nn.generate_ps_assemblies('gen_ordered')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()
    nn.set_conductance_monitor()

    '''
    gr0 = nn.p_ass_index[0][0]
    t = 21*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 21.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    '''

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 
    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(4*second, .01)
    nn.mon_spike_e.record = True
    nn.mon_spike_i.record = True
    nn.balance(1*second, .01)

    nstim = 20
    currs = [10*pA, 20*pA, 40*pA, 80*pA, 150*pA]
    dur_stim, dur_relx = 100, 400
    dur = dur_stim + dur_relx 

    for curr in currs:
        for i in range(nstim):
            nn.stim_curr(curr=curr, dur_stim=dur_stim, dur_relx=dur_relx)

    plotter.plot_pop_raster(nn)

    nsubs = len(currs)
    mfrl = []
    wbin = .1
    dur_stim, dur_pre = 120, 20
    base_fr = 5.
    plt.figure()
    for i, curr in enumerate(currs):
        tl = 20000 + i*nstim*dur + np.arange(nstim)*dur
        plt.subplot(nsubs, 1, 1+i)
        mfr = plotter.plot_mean_curr_act(nn, tl, dur_stim=dur_stim,
                                        dur_pre=dur_pre, wbin=wbin)
        mfrl.append(calc_spikes.gaus_smooth(mfr, w=wbin, sigma=.2))

        peak_time = np.argmax(mfrl[-1])*wbin - dur_pre
        peak_value = np.max(mfrl[-1])
        
        peak80_time = (mfr > base_fr+(.8*(peak_value-base_fr))).argmax()*wbin - dur_pre
        peak20_time = (mfr > base_fr+(.2*(peak_value-base_fr))).argmax()*wbin - dur_pre

        time_const = peak80_time - peak20_time 
        print 'time const is ', time_const 

    plt.show()
    return nn


def test_boost_pf():
    Ne=20000
    gfc = 1
    pr = 0.1
    #pf= 0.06/gfc
    pf = 0.00
    pf_boost = 0.04

    Ni = Ne/4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1*nS
    gi0 = 0.4*nS
 
    #gee= ge0
    gee = ge0*(20000./Ne)**.5
    gii = gi0*(20000./Ne)**.5


    pf = pf*(Ne/20000.)**.5
    pr = pr*(Ne/20000.)**.5

    nn = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
        n_ass=10, s_ass=500, pr=pr, pf=pf, ext_input=200*pA,
        g_ee=gee, g_ie=gee, g_ei=gii, g_ii=gii, g_ff_coef=gfc,
        modified_contin=True)

    #nn.generate_ps_assemblies('gen_no_overlap')
    nn.generate_ps_assemblies('gen_ordered')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()

    #return nn

    gr = nn.p_ass_index[0][0]
    '''
    '''
    t = 19.5*second
    nn.set_noisy_input(gr, t, sigma=0*ms) 
    t = 20*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    t = 22*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    t = 24*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    t = 26*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    t = 28*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    for i in range(9):
        t = (29+i)*second
        nn.set_noisy_input(gr, t, sigma=0*ms)

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 

    #nn.boost_pff(0.04)
    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(4*second, .01)
    nn.mon_spike_e.record = True 
    nn.mon_spike_i.record = True 
    nn.balance(1*second, .01)
    nn.boost_pff(pf_boost)
    nn.balance(2*second, 5.)
    nn.balance(2*second, 1.)
    nn.balance(2*second, .1)
    nn.balance(2*second, .01)
    nn.run_sim(4*second)

    return nn


def test_boost_pf_cont():
    Ne=20000
    gfc = 1
    pr = 0.08
    #pf= 0.06/gfc
    pf = 0.00
    pf_boost = 0.04

    Ni = Ne/4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1*nS
    gi0 = 0.4*nS
 
    #gee= ge0
    gee = ge0*(20000./Ne)**.5
    gii = gi0*(20000./Ne)**.5


    pf = pf*(Ne/20000.)**.5
    pr = pr*(Ne/20000.)**.5

    nn = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
        n_ass=10, s_ass=500, pr=pr, pf=pf, ext_input=200*pA,
        g_ee=gee, g_ie=gee, g_ei=gii, g_ii=gii, g_ff_coef=gfc, 
        continuous_ass=True, modified_contin=True)

    #nn.generate_ps_assemblies('gen_no_overlap')
    nn.generate_ps_assemblies('gen_ordered')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()

    #return nn

    gr = nn.p_ass_index[0][0]
    '''
    '''
    t = 19.*second
    nn.set_noisy_input(gr, t, sigma=0*ms) 
    t = 19.5*second
    #nn.set_noisy_input(gr, t, sigma=0*ms)
    #t = 22*second
    #nn.set_noisy_input(gr, t, sigma=0*ms)
    #t = 24*second
    #nn.set_noisy_input(gr, t, sigma=0*ms)
    #t = 26*second
    #nn.set_noisy_input(gr, t, sigma=0*ms)
    #t = 28*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    for i in range(9):
        t = (28+i)*second
        nn.set_noisy_input(gr, t, sigma=0*ms)

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 

    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(4*second, .01)
    nn.mon_spike_e.record = True 
    nn.mon_spike_i.record = True 
    nn.balance(1*second, .01)
    nn.boost_pff(pf_boost)
    nn.balance(2*second, 5.)
    nn.balance(2*second, 1.)
    nn.balance(2*second, .1)
    nn.balance(2*second, .01)
    nn.run_sim(9*second)

    frac = .1
    fname = 'contASS_pr' + str(pr)+ 'pfboost' + str(pf_boost) + \
            'frac' + str(frac) + '.npz'
    spikes4save = calc_spikes.get_spike_times_ps(nn, 
                    frac=frac, pick_first=False)
    np.savez_compressed(fname, spikes4save)

    return nn


def test_slopes():
    nn=Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
        n_ass=1, s_ass=500, pr=.0, pf=.0,
        ext_input=200*pA,
        g_ee=0.1*nS, g_ie=0.1*nS, g_ei=0.4*nS, g_ii=0.4*nS
        )
    nn.generate_ps_assemblies('gen_ordered')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()
    nn.set_conductance_monitor()

    '''
    gr0 = nn.p_ass_index[0][0]
    t = 21*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 21.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 22.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    t = 23.5*second
    nn.set_noisy_input(gr0, t, sigma=0*ms) 
    '''

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 
    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(4*second, .01)
    nn.mon_spike_e.record = True
    nn.mon_spike_i.record = True
    nn.balance(1*second, .01)

    for nrn in nn.p_ass_index[0][0]:
        nn.Pe[nrn].I += 5*pA
    nn.run_sim(.5*second)
    for nrn in nn.p_ass_index[0][0]:
        nn.Pe[nrn].I -= 5*pA
    nn.run_sim(.5*second)

    for nrn in nn.p_assinh_index[0][0]:
        nn.Pi[nrn].I += 5*pA
    nn.run_sim(.5*second)
    for nrn in nn.p_assinh_index[0][0]:
        nn.Pi[nrn].I -= 5*pA
    nn.run_sim(.5*second)


    fe = calc_spikes.make_fr_from_spikes(nn, 0, 5, True)[0]
    fi = calc_spikes.make_fr_from_spikes(nn, 0, 5, False)[0] 
    plt.subplot(211)
    plt.plot(fe)
    plt.subplot(212)
    plt.plot(fi)

    #plt.show()
    return nn


def test_contin(Ne=20000):
    gfc = 1
    pr = 0.06
    #pf= 0.06/gfc
    pf = 0.06

    #Ne=20000
    Ni = Ne/4
    cp = .01

    # the default conductances used for Ne=20000
    ge0 = 0.1*nS
    gi0 = 0.4*nS
 
    #gee= ge0
    gee = ge0
    gii = gi0

    '''
    s_ass = 250
    pr = .12
    pf = .12

    '''
    n_ass = 10
    s_ass = 500
    pr = .06
    pf = .06

    '''
    n_ass = 10
    s_ass = 50
    pr = .6
    pf = .6
    '''

    continuous_ass = True
    nn = Nets(Ne=Ne, Ni=Ni, cp_ee=cp, cp_ie=cp, cp_ei=cp, cp_ii=cp,
        n_ass=n_ass, s_ass=s_ass, pr=pr, pf=pf, ext_input=200*pA,
        g_ee=gee, g_ie=gee, g_ei=gii, g_ii=gii, g_ff_coef=gfc,
        continuous_ass=continuous_ass)

    #nn.generate_ps_assemblies('gen_no_overlap')
    nn.generate_ps_assemblies('gen_ordered')
    nn.set_net_connectivity()
    nn.set_spike_monitor()
    nn.set_rate_monitor()
    nn.set_voltage_monitor()
    nn.set_current_monitor()

    #return nn

    gr = nn.p_ass_index[0][0]
    '''
    '''
    #t = 20*second
    #nn.set_noisy_input(gr, t, sigma=0*ms) 
    t = 20*second
    nn.set_noisy_input(gr, t, sigma=0*ms) 
    t = 21*second
    nn.set_noisy_input(gr, t, sigma=0*ms) 
    t = 22*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    t = 23*second
    nn.set_noisy_input(gr, t, sigma=0*ms)
    t = 24*second
    nn.set_noisy_input(gr, t, sigma=0*ms)

    nn.mon_spike_e.record = False 
    nn.mon_spike_i.record = False 

    '''
    nn.balance(20*second,5.)
    nn.balance(20*second,1.)
    nn.balance(10*second,.1)
    nn.balance(10*second,.01)
    '''
    #nn.balance(.5*second, 5.)
    #return nn

    nn.balance(5*second, 5.)
    nn.balance(5*second, 1.)
    nn.balance(5*second, .1)
    nn.balance(5*second, .01)
    nn.mon_spike_e.record = True 
    nn.mon_spike_i.record = True 
    nn.run_sim(5*second)

    '''
    t_end = nn.network.clock.t*1000/second
    figure = plt.figure(figsize=(16., 12.))
    plotter.plot_ps_raster(nn, chain_n=0, frac=1., figure=figure,
                            dummy_ass=True)
    plt.xlim([t_end-2000, t_end])
    '''
    #fname_pre = 'large_nets/scaleAllg_Ne'
    #plt.savefig(fname_pre + str(Ne) + 'pr' + str(pr) + 'pf' + str(pf) + 

    #figure2 = plt.figure(figsize=(16.,12.))
    #plt.plot(nn.mon_rate_e.smooth_rate(5*ms))
    #plt.xlim([5000,t_end])
    #plt.xlabel('t [ms]')
    #plt.ylabel('FR [sp/sec]')
    #plt.savefig(fname_pre+ str(Ne) + 'pr'+str(pr)+'pf'+str(pf) + '_fr'+'.png')

    #plt.close('all')
    #plotter.plot_fr_cv_syn_distr(nn)
    return nn


if __name__=='__main__':
    from matplotlib import pyplot as plt

    '''
    nn=Nets(Ne=20000, Ni=5000, cp_ee=.01, cp_ie=.01, cp_ei=0.01, cp_ii=.01,
        n_ass=444, s_ass=150, pr=.2, pf=.2, synapses_per_nrn=200,
        #n_ass=10,s_ass=500,pr=.15,pf=.03,symmetric_sequence=True,p_rev=.03,
        #n_ass=10,s_ass=500,pr=.06,pf=.06,
        g_ee=0.1*nS, g_ie=0.1*nS, g_ei=0.4*nS, g_ii=0.4*nS)
    nn.plot_for_raster_curr_volt()

    ie= float(sys.argv[1])
    ii= float(sys.argv[2])
    tr= float(sys.argv[3])
    nn.test_shifts(ie,ii,tr)
    '''

    '''
    ne= int(sys.argv[1])
    nn = test_diff_gff(ne)
    '''
    #nn = test_psps()

    #nn = test_symm()
    #nn = test_fr()
    #nn = test_noPS()

    #nn = test_diff_gff()
    #nn = test_longseq()
    #nn = test_2ASS()
    #nn = show_ass_frs()
    #nn = test_tau()
    #nn = test_boost_pf()
    #nn = test_slopes()
    #nn = test_boost_pf_cont()
    #nn = test_boost_pf()
    #nn = test_contin()


    nn = show_ass_frs()

    plt.figure()
    plotter.plot_pop_raster(nn)
    plt.ylim([0, 12*nn.s_ass])
    plt.xlim([22950, 23150])
    plotter.show()



