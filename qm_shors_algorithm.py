from MPS import *
import Gates as gates
import numpy as np
from MPO import *
from MPSU import *
import matplotlib.pyplot as plt
import shor_helpers as sh
import Circuits as circ
import time
import tim 

cutoff = 1e-8
def get_shor_fat(N,x,xi):
    tim1 = tim.Tim()
    
    len_b = int(np.ceil(np.log2(N)))
    len_a = 2*len_b+3

    # Create the MPSU
    mpsu = MPSU.create_MPSU_init_to_1(len_a,len_b,xi,cutoff)
    tim1.print_since_last("MPSU created in")
    #apply hadamards gates to all sites of len_a
    for i in range(len_a):
        mpsu.apply_1_site_gate(gates.H,i)
    
    tim1.print_since_last("Hadamards applied in")

    #get control U gates
    cu_gates = gates.cx_pow_2k_mod_N(N,x,len_a)

    tim1.print_since_last("Control U gates created in")

    tim2 = tim.Tim()
    #apply the controlled U gates
    for i in range(len_a):
        mpsu.apply_fat_xgate(cu_gates[len_a-1-i])
        tim2.print_since_last("Control U "+str(i)+" gate applied in")
        for j in range(len_a-2,i-1,-1):
            mpsu.apply_2_site_gate(gates.SWAP,j)
        tim2.print_since_last("SWAP gates applied in")
    
    tim1.print_since_start("Total time for applying gates all fat U gates")
    tim1.print_since_start("Total time before inverse Fourier transform")
    return mpsu , len_a, tim1

def get_shor_mpo(N,x,xi,mpos):
    tim1 = tim.Tim()
    len_b = int(np.ceil(np.log2(N)))
    len_a = 2*len_b+3
    #print(len_a,len_b,"len_a,len_b")
    len_total = len_a+len_b

    # Create the MPS
    mps = MPS.create_MPS_init_to_N(1,len_total,xi=xi,cutoff=cutoff)

    #apply hadamards gates to all sites of len_a
    for i in range(len_a):
        mps.apply_1_site_gate(gates.H,i)

    tim1.print_since_last("MPS initialized in")

    #apply the controlled U gates
    for i in range(len_a):
        tim2 = tim.Tim()
        mps.apply_mpo_zip_up_2(mpos[len_a-1-i],len_a-1)
        tim2.print_since_last("MPO applied in")
        for j in range(len_a-2,i-1,-1):
            mps.apply_2_site_gate(gates.SWAP,j) 
        tim2.print_since_last("SWAP gates applied in")
    
    tim1.print_since_last("Control U gates applied in")
    tim1.print_since_start("Total time before inverse Fourier transform")
    return mps, len_a, tim1




