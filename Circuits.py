'''Circuits.py'''

from MPS import MPS
from Gates import *

def reverse_bit_order(MPS):
    for i in range(MPS.L-1):
        for j in range(MPS.L-1-i):
            print(j,"j")
            MPS.apply_2_site_gate(SWAP,j)
    return MPS

#Fourier transform MPS
def fourier_transform_MPS(MPS,i,j,inv=False,rev=False):
    for n in range(i,j):
        MPS.apply_1_site_gate(H,0)
        for m in range(0,j-1-n):
            mgate = combine(SWAP,R(m+2,inv=inv))
            MPS.apply_2_site_gate(mgate,m)
    if rev:
        MPS = reverse_bit_order(MPS)
    return MPS

