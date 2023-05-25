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

def get_shor_mps_mpo(x,N,xi):
    tim1 = tim.Timer()
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
    #get control U gates
    cu_gates = gates.cx_pow_2k_mod_N(N,x,len_a)
    tim1.print_since_last("Control U gates created in")
    mpo = 0

    #apply the controlled U gates
    for i in range(len_a):
        tim2 = tim.Timer()
        mpo = MPO.create_MPO_from_tensor(cu_gates[len_a-1-i],xi,cutoff=cutoff)
        tim2.print_since_last("MPO created in")
        mps.apply_mpo_regularily(mpo,len_a-1)
        tim2.print_since_last("MPO applied in")
        for j in range(len_a-2,i-1,-1):
            mps.apply_2_site_gate(gates.SWAP,j) 
        tim2.print_since_last("SWAP gates applied in")
    
    tim1.print_since_last("Control U gates applied in")
    first_mpo = mpo

    #apply the inverse fourier transform
    mps = circ.fourier_transform_MPS(mps,0,len_a,inv=True)
    tim1.print_since_last("Inverse Fourier transform applied in")
    return mps, len_a, first_mpo

def get_shor_mps_fat_ten(x,N,xi):
    timer = tim.Timer()
    len_b = int(np.ceil(np.log2(N)))
    len_a = 2*len_b+3
    #print("len_a",len_a,"len_b",len_b)

    # Create the MPSU
    mpsu = MPSU.create_MPSU_init_to_1(len_a,len_b,xi,cutoff)
    timer.print_since_last("MPSU created in")
    #apply hadamards gates to all sites of len_a
    for i in range(len_a):
        mpsu.apply_1_site_gate(gates.H,i)
    
    timer.print_since_last("Hadamards applied in")

    #get control U gates
    cu_gates = gates.cx_pow_2k_mod_N(N,x,len_a)

    timer.print_since_last("Control U gates created in")

    timer2 = tim.Timer()
    #apply the controlled U gates
    for i in range(len_a):
        mpsu.apply_fat_xgate(cu_gates[len_a-1-i])
        timer.print_since_last("Control U gate applied in")
        for j in range(len_a-2,i-1,-1):
            mpsu.apply_2_site_gate(gates.SWAP,j)
        timer.print_since_last("SWAP gates applied in")
    timer2.print_since_start("Total time for applying gates all fat U gates")
    
    #apply the inverse fourier transform
    mpsu = circ.fourier_transform_MPS(mpsu,0,len_a,inv=True)
    timer.print_since_last("Inverse Fourier transform applied in")
    timer.print_since_start("Total time")
    return mpsu, len_a