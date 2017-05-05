prc = 0.06
pff = 0.06
prand = 0.01
N = 20000
M = 500
L = 10



f=1
fl = [1,4,8,9,10,16,25]

for f in fl:

    # #memory synapses per nrn
    ps_syn_nrn = (prc+pff) * f**.5 * M
    # # random nrns per neuron
    rand_syn_nrn = f*prand*N
    syn_ratio_nrn = ps_syn_nrn / (rand_syn_nrn+ps_syn_nrn)

    # # memory connections
    ps_syn_all = (L*prc + (L-1)*pff) * f**.5 * M**2
    # # random connections
    rand_syn_all = prand*N**2*f
    syn_ratio_all = ps_syn_all / (rand_syn_all+ps_syn_all)
    #print f, ps_syn_nrn, rand_syn_nrn
    #print f, ps_syn_all, rand_syn_all
    print f, syn_ratio_nrn, syn_ratio_all

