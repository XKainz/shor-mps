'''Gate.py'''
import numpy as np
import numpy.linalg as la
import cmath
import math

def combine(gate1,gate2):
    if gate1.shape != gate2.shape:
        raise ValueError("Gates must have the same shape")
    D = len(gate1.shape)//2
    return np.tensordot(gate1,gate2,axes=([*range(D,2*D)],[*range(0,D)]))
    
H = np.array([[1,1],[1,-1]]).reshape(2,2)/np.sqrt(2)
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]).reshape(2,2,2,2)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).reshape(2,2,2,2)
def R(k,inv=False): 
    return np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,cmath.exp((-1)**inv * 1j*2*math.pi/(2**k))]]).reshape([2,2,2,2])

def x_mod_N(N,x):
    L = int(np.ceil(np.log2(N)))
    U = np.zeros((2**L,2**L))
    for i in range(2**L):
        if(i<N):
            U[(i*x)%N,i] = 1
        else:
            U[i,i]=1
    U = U.reshape([2]*2*L)
    return U

def cx_mod_N(N,x):
    x_mod_N = x_mod_N(N,x)
    d = len(x_mod_N.shape)//2
    x_mod_N = np.reshape(x_mod_N,(2**d,2**d))
    cx_mod_N = np.zeros((2**(d+1),2**(d+1)))
    cx_mod_N[:2**d,:2**d] = np.identity(2**d)
    cx_mod_N[2**d:,2**d:] = x_mod_N
    cx_mod_N = cx_mod_N.reshape([2]*(d+1)*2)
    return cx_mod_N
                        
def cx_pow_2k_mod_N(N,x,k):
    cx_pow_2k_mod_N = [cx_mod_N(N,x)]
    for i in range(1,k):
        cx_pow_2k_mod_N.append(combine(cx_pow_2k_mod_N[i-1],cx_pow_2k_mod_N[i-1]))
    return cx_pow_2k_mod_N

def rand_1_site_U():
    phi = np.random.uniform(0,2*np.pi)
    v = np.random.rand(2)+1j*np.random.rand(2)
    v = v/la.norm(v)
    U = np.array([[v[0],v[1]],
                 [-np.conj(v[1])*np.exp(1j*phi),np.exp(1j*phi)*np.conj(v[0])]])
    return U

