'''Circuits.py'''

from MPS import MPS
from MPO import *
from Gates import *
import tim

def reverse_bit_order(MPS):
    for i in range(MPS.L-1):
        for j in range(MPS.L-1-i):
            MPS.apply_2_site_gate(SWAP,j)
    return MPS

def reverse_bit_order_mpo(mpo):
    for i in range(mpo.L-1):
        for j in range(mpo.L-1-i):
            mpo.merge_2_site_gate(SWAP,j)
    return mpo

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

def get_delta3():
    delta3 = np.zeros((2,)*3)
    delta3[0,0,0] = 1
    delta3[1,1,1] = 1
    return delta3

def get_3_phase(phi):
    phase3 = np.zeros((2,)*3,dtype="complex")
    phase3[0,:,:]=np.eye(2)
    phase3[1,:,:]=np.array([[1,0],[0,np.exp(1j*phi)]])
    return phase3.reshape(2,2,2,1)

def get_4_phase(phi):
    phase3 = get_3_phase(phi).reshape(2,2,2)
    phase4 = np.einsum("ijk,inm->njkm",phase3,get_delta3())
    return phase4

def get_H_copy():
    return np.einsum("ijk,jl->ilk",get_delta3(),H).reshape(1,2,2,2)

#Build Fourier transform MPO
def get_H_Phase_alist(len_a):
    alist = [get_H_copy().reshape(1,2,2,2)]
    for i in range(1,len_a-1):
        alist.append(get_4_phase(math.pi/2**(i)))
    alist.append(get_3_phase(math.pi/2**(len_a-1)))
    return alist

def get_fourier_transform_mpo(len_a,xi=2**8):
    MPOlist = [np.ones(1)]
    for i in range(len_a):
        MPOlist.append(np.eye(2).reshape(1,2,2,1))
        MPOlist.append(np.ones(1)) 
    mpo = MPO.create_MPO_from_tensor_array(MPOlist,xi=xi,cutoff=1e-8)
    for i in range(len_a-1):
        alist = get_H_Phase_alist(len_a-i)
        mpo.merge_mpo_zip_up_inv_alist(alist,i)
    mpo.merge_1_site_gate(H,len_a-1)
    return mpo

##Move this to MPO.py
def get_identity_mpo(len_a,xi=2**8):
    MPOlist = [np.ones(1)]
    for i in range(len_a):
        MPOlist.append(np.eye(2).reshape(1,2,2,1))
        MPOlist.append(np.ones(1)) 
    mpo = MPO.create_MPO_from_tensor_array(MPOlist,xi=xi,cutoff=1e-8)
    return mpo
