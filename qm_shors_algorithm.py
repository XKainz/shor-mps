import MPS as mps
import Gates as gates
import numpy as np
import MPO as mpo
import matplotlib.pyplot as plt
import shor_helpers as sh
import Circuits as circ
import time

cutoff = 1e-8

def get_shor_mps_mpo(x,N,xi):
    t0 = time.time()
    len_b = int(np.ceil(np.log2(N)))
    len_a = 2*len_b+3
    print(len_a,len_b,"len_a,len_b")
    len_total = len_a+len_b

    # Create the MPS
    MPS = mps.MPS.create_MPS_init_to_N(1,len_total,xi=xi,cutoff=cutoff)

    #apply hadamards gates to all sites of len_a
    for i in range(len_a):
        MPS.apply_1_site_gate(gates.H,i)

    #get control U gates
    cu_gates = gates.cx_pow_2k_mod_N(N,x,len_a)

    #apply the controlled U gates
    for i in range(len_a):
        mpocu = mpo.MPO(cu_gates[len_a-1-i],xi=xi,cutoff=cutoff)
        MPS.apply_mpo(mpocu,len_a-1)
        for j in range(len_a-2,i-1,-1):
            MPS.apply_2_site_gate(gates.SWAP,j) 

    #apply the inverse fourier transform
    MPS = circ.fourier_transform_MPS(MPS,0,len_a,inv=True)
    t1 = time.time()
    print("MPS created in",t1-t0,"seconds")
    return MPS, len_a